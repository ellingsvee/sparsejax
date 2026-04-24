from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
import numpy as np

from sparsejax.sparse import SparseMatrix


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
    rows = (uniq // n_cols).astype(np.int32)
    cols = (uniq % n_cols).astype(np.int32)
    c_indices = np.stack([rows, cols], axis=0)
    return c_indices, a_to_c.astype(np.int64), b_to_c.astype(np.int64)


def spadd(A: SparseMatrix, B: SparseMatrix) -> SparseMatrix:
    if A.shape != B.shape:
        raise ValueError(f"shape mismatch: A {A.shape}, B {B.shape}")
    c_indices, a_to_c, b_to_c = _union_pattern(A.indices, B.indices, A.shape)
    nnz_c = c_indices.shape[1]
    out_dtype = jnp.result_type(A.data.dtype, B.data.dtype)
    out = jnp.zeros(nnz_c, dtype=out_dtype)
    out = out.at[a_to_c].add(A.data.astype(out_dtype))
    out = out.at[b_to_c].add(B.data.astype(out_dtype))
    return SparseMatrix(data=out, indices=c_indices, shape=A.shape)
