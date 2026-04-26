from __future__ import annotations

from typing import Tuple
import weakref

import jax.numpy as jnp
import numpy as np

from sparsejax.sparse import SparseMatrix, SparseStructure


class _UnionCacheEntry:
    __slots__ = ("structure", "a_to_c", "b_to_c", "a_ref", "b_ref")

    def __init__(self, structure, a_to_c, b_to_c, a_ref, b_ref):
        self.structure = structure
        self.a_to_c = a_to_c
        self.b_to_c = b_to_c
        self.a_ref = a_ref
        self.b_ref = b_ref


_UNION_CACHE: dict[tuple[int, int, tuple[int, int]], _UnionCacheEntry] = {}


def _drop_union_cache(key) -> None:
    _UNION_CACHE.pop(key, None)


def _output_index_dtype(*arrays: np.ndarray, shape: Tuple[int, int]) -> np.dtype:
    if max(shape) > np.iinfo(np.int32).max:
        return np.dtype(np.int64)
    if any(np.asarray(a).dtype == np.dtype(np.int64) for a in arrays):
        return np.dtype(np.int64)
    return np.dtype(np.int32)


def _union_pattern(
    a_indices: np.ndarray,
    b_indices: np.ndarray,
    shape: Tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (c_indices, a_to_c, b_to_c): indices of the union and the
    mappings from each input's nnz to the union position."""
    n_cols = shape[1]
    a_lin = a_indices[0].astype(np.int64) * n_cols + a_indices[1].astype(np.int64)
    b_lin = b_indices[0].astype(np.int64) * n_cols + b_indices[1].astype(np.int64)
    combined = np.concatenate([a_lin, b_lin])
    uniq, inv = np.unique(combined, return_inverse=True)
    a_to_c = inv[: a_lin.shape[0]]
    b_to_c = inv[a_lin.shape[0] :]
    index_dtype = _output_index_dtype(a_indices, b_indices, shape=shape)
    rows = (uniq // n_cols).astype(index_dtype)
    cols = (uniq % n_cols).astype(index_dtype)
    c_indices = np.stack([rows, cols], axis=0)
    return c_indices, a_to_c.astype(np.int64), b_to_c.astype(np.int64)


def _cached_union_pattern(
    a_indices: np.ndarray,
    b_indices: np.ndarray,
    shape: Tuple[int, int],
) -> tuple[SparseStructure, np.ndarray, np.ndarray]:
    key = (id(a_indices), id(b_indices), tuple(shape))
    entry = _UNION_CACHE.get(key)
    if entry is not None:
        a_cached = entry.a_ref() if entry.a_ref is not None else a_indices
        b_cached = entry.b_ref() if entry.b_ref is not None else b_indices
        if a_cached is a_indices and b_cached is b_indices:
            return entry.structure, entry.a_to_c, entry.b_to_c
        _UNION_CACHE.pop(key, None)

    c_indices, a_to_c, b_to_c = _union_pattern(a_indices, b_indices, shape)
    structure = SparseStructure(indices=c_indices, shape=shape)
    try:
        a_ref = weakref.ref(a_indices)
        weakref.finalize(a_indices, _drop_union_cache, key)
    except TypeError:
        a_ref = None
    try:
        b_ref = weakref.ref(b_indices)
        if b_indices is not a_indices:
            weakref.finalize(b_indices, _drop_union_cache, key)
    except TypeError:
        b_ref = None
    _UNION_CACHE[key] = _UnionCacheEntry(structure, a_to_c, b_to_c, a_ref, b_ref)
    return structure, a_to_c, b_to_c


def spadd(A: SparseMatrix, B: SparseMatrix) -> SparseMatrix:
    if A.shape != B.shape:
        raise ValueError(f"shape mismatch: A {A.shape}, B {B.shape}")
    if A.structure == B.structure:
        return SparseMatrix(data=A.data + B.data, structure=A.structure)
    c_structure, a_to_c, b_to_c = _cached_union_pattern(A.indices, B.indices, A.shape)
    nnz_c = c_structure.nnz
    out_dtype = jnp.result_type(A.data.dtype, B.data.dtype)
    out = jnp.zeros(nnz_c, dtype=out_dtype)
    out = out.at[a_to_c].add(A.data.astype(out_dtype))
    out = out.at[b_to_c].add(B.data.astype(out_dtype))
    return SparseMatrix(data=out, structure=c_structure)
