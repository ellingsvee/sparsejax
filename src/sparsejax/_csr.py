"""COO to CSR conversion

Also ensure the factor token is stable. This is a process-unique integer that serves as a key for caching the symbolic factorization.
"""

from __future__ import annotations

from typing import NamedTuple, Tuple
import threading
import weakref

import numpy as np


class CsrStructure(NamedTuple):
    indptr: np.ndarray
    col_idx: np.ndarray
    order: np.ndarray
    order_is_identity: bool
    shape: Tuple[int, int]
    factor_token: int
    indices_ref: "weakref.ReferenceType[np.ndarray] | None"


_CACHE: dict[tuple[int, str], CsrStructure] = {}
_TOKEN_LOCK = threading.Lock()
_NEXT_TOKEN = 1


def _alloc_token() -> int:
    global _NEXT_TOKEN
    with _TOKEN_LOCK:
        tok = _NEXT_TOKEN
        _NEXT_TOKEN += 1
    return tok


def _native_drop_token(token: int) -> None:
    """Best-effort free of the cached cuDSS state for `token`."""
    try:
        from sparsejax import backend as _native  # type: ignore
    except ImportError:
        return
    drop = getattr(_native, "cudss_drop_token", None)
    if drop is not None:
        try:
            drop(int(token))
        except Exception:
            pass


def _on_indices_collected(key: tuple[int, str], token: int) -> None:
    _CACHE.pop(key, None)
    _native_drop_token(token)


def _normalize_index_dtype(index_dtype) -> np.dtype:
    dt = np.dtype(index_dtype)
    if dt not in (np.dtype(np.int32), np.dtype(np.int64)):
        raise TypeError(f"index_dtype must be int32 or int64, got {dt}")
    return dt


def _pick_index_dtype(
    indices: np.ndarray, shape: Tuple[int, int], nnz: int
) -> np.dtype:
    max_i32 = np.iinfo(np.int32).max
    if indices.dtype == np.dtype(np.int64):
        return np.dtype(np.int64)
    if nnz > max_i32 or max(shape) > max_i32:
        return np.dtype(np.int64)
    return np.dtype(np.int32)


def _check_index_capacity(
    indices: np.ndarray,
    shape: Tuple[int, int],
    nnz: int,
    index_dtype: np.dtype,
) -> None:
    if indices.size and np.min(indices) < 0:
        raise ValueError("sparse indices must be non-negative")
    if index_dtype != np.dtype(np.int32):
        return
    max_i32 = np.iinfo(np.int32).max
    if nnz > max_i32:
        raise OverflowError(
            f"nnz={nnz} exceeds int32 sparse index capacity; use int64 indices"
        )
    if max(shape) > max_i32:
        raise OverflowError(
            f"shape={shape} exceeds int32 sparse index capacity; use int64 indices"
        )
    if indices.size and np.max(indices) > max_i32:
        raise OverflowError(
            "sparse index value exceeds int32 capacity; use int64 indices"
        )


def coo_to_csr(
    indices: np.ndarray,
    shape: Tuple[int, int],
    *,
    index_dtype=None,
    upper: bool = False,
) -> CsrStructure:
    if upper and shape[0] != shape[1]:
        raise ValueError(f"upper CSR view requires a square shape, got {shape}")
    dt = (
        _pick_index_dtype(indices, shape, indices.shape[1])
        if index_dtype is None
        else _normalize_index_dtype(index_dtype)
    )
    _check_index_capacity(indices, shape, indices.shape[1], dt)

    key = (id(indices), dt.str, "upper" if upper else "full")
    entry = _CACHE.get(key)
    if entry is not None and entry.shape == shape:
        # Reject stale entries left behind by id reuse
        cached = entry.indices_ref() if entry.indices_ref is not None else None
        if entry.indices_ref is None or cached is indices:
            return entry
    if entry is not None:
        # Stale entry — drop the cached cuDSS state too before rebuilding.
        _native_drop_token(entry.factor_token)
        _CACHE.pop(key, None)
    row = np.asarray(indices[0], dtype=np.int64)
    col = np.asarray(indices[1], dtype=np.int64)
    if upper:
        active = row <= col
        row = row[active]
        col = col[active]
        original_pos = np.nonzero(active)[0]
    else:
        original_pos = None
    n_rows = shape[0]
    # Sort primarily by row, secondarily by col — gives canonical CSR layout.
    order = np.lexsort((col, row))
    final_order = order if original_pos is None else original_pos[order]
    order_is_identity = bool(
        final_order.size == indices.shape[1]
        and (
            final_order.size == 0
            or np.array_equal(final_order, np.arange(final_order.size))
        )
    )
    row_sorted = row[order]
    col_sorted = col[order]
    counts = np.bincount(row_sorted, minlength=n_rows)
    indptr = np.empty(n_rows + 1, dtype=dt)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    token = _alloc_token()
    try:
        indices_ref: "weakref.ReferenceType[np.ndarray] | None" = weakref.ref(indices)
    except TypeError:
        indices_ref = None
    csr = CsrStructure(
        indptr=indptr,
        col_idx=col_sorted.astype(dt, copy=False),
        order=final_order,
        order_is_identity=order_is_identity,
        shape=shape,
        factor_token=token,
        indices_ref=indices_ref,
    )
    _CACHE[key] = csr
    try:
        weakref.finalize(indices, _on_indices_collected, key, token)
    except TypeError:
        pass
    return csr
