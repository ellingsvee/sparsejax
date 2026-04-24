from __future__ import annotations


import jax.numpy as jnp
import numpy as np

from sparsejax.sparse import SparseMatrix
from sparsejax.utils import _resolve_backend


def _dispatch_spspmm(
    backend_name: str,
    a_data,
    a_row,
    a_col,
    a_shape,
    b_data,
    b_row,
    b_col,
    b_shape,
):
    if backend_name in ("scipy", "cholmod"):
        from sparsejax.backends import scipy_backend

        return scipy_backend.spspmm(
            a_data,
            a_row,
            a_col,
            a_shape,
            b_data,
            b_row,
            b_col,
            b_shape,
        )
    if backend_name in ("cudss", "cudss_ffi"):
        # cuDSS itself doesn't expose SpGEMM; the GPU path is via cuSPARSE
        # (cupy). Fall back to scipy on host if cupy isn't installed — the
        # cost is bounded by SpGEMM not being jit-friendly anyway.
        try:
            from sparsejax.backends import cudss_backend

            return cudss_backend.spspmm(
                a_data,
                a_row,
                a_col,
                a_shape,
                b_data,
                b_row,
                b_col,
                b_shape,
            )
        except (ImportError, RuntimeError):
            from sparsejax.backends import scipy_backend

            return scipy_backend.spspmm(
                a_data,
                a_row,
                a_col,
                a_shape,
                b_data,
                b_row,
                b_col,
                b_shape,
            )
    raise ValueError(f"unknown spspmm backend: {backend_name!r}")


def spspmm(
    A: SparseMatrix,
    B: SparseMatrix,
    *,
    backend: str | None = "auto",
) -> SparseMatrix:
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"shape mismatch: A {A.shape} @ B {B.shape}")
    backend_name = _resolve_backend(A, backend)
    out_dtype = jnp.result_type(A.data.dtype, B.data.dtype)
    c_data, c_row, c_col, _ = _dispatch_spspmm(
        backend_name,
        np.asarray(A.data),
        A.indices[0],
        A.indices[1],
        A.shape,
        np.asarray(B.data),
        B.indices[0],
        B.indices[1],
        B.shape,
    )
    c_indices = np.stack(
        [np.asarray(c_row, dtype=np.int32), np.asarray(c_col, dtype=np.int32)],
        axis=0,
    )
    return SparseMatrix(
        data=jnp.asarray(c_data, dtype=out_dtype),
        indices=c_indices,
        shape=(A.shape[0], B.shape[1]),
    )
