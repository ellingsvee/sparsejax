from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from sparsejax.sparse import SparseMatrix
from sparsejax._csr import coo_to_csr


def _spdmm_via_ffi(
    data: jax.Array,
    indices: np.ndarray,
    shape: tuple,
    X: jax.Array,
    *,
    transpose: bool,
) -> jax.Array:
    """Sparse-dense matmul as an XLA-lowered primitive.

    The primitive itself dispatches to the right per-platform FFI handler
    (cuSPARSE on CUDA, cpu_csr_sparse_dense_ffi on CPU) at MLIR-lowering
    time, so the same call site works under jit on either device without
    the caller having to know the device platform.
    """
    from sparsejax.backends import _cusparse_lowering as cusp

    csr = coo_to_csr(indices, shape)
    data_csr = data[csr.order]
    indptr = jnp.asarray(csr.indptr, dtype=jnp.int32)
    col_idx = jnp.asarray(csr.col_idx, dtype=jnp.int32)
    if X.ndim == 1:
        return cusp.csr_matvec(
            data_csr,
            col_idx,
            indptr,
            X,
            shape=shape,
            transpose=transpose,
        )
    return cusp.csr_matmat(
        data_csr,
        col_idx,
        indptr,
        X,
        shape=shape,
        transpose=transpose,
    )


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4))
def _spdmm_impl(
    data: jax.Array,
    X: jax.Array,
    indices: np.ndarray,
    shape: tuple,
    transpose: bool,
) -> jax.Array:
    return _spdmm_via_ffi(data, indices, shape, X, transpose=transpose)


def _spdmm_fwd(data, X, indices, shape, transpose):
    Y = _spdmm_impl(data, X, indices, shape, transpose)
    return Y, (data, X)


def _spdmm_bwd(indices, shape, transpose, residuals, dY):
    data, X = residuals
    # ∂X: forward was A @ X (or Aᵀ @ X). Cotangent is Aᵀ @ dY (or A @ dY).
    dX = _spdmm_impl(data, dY, indices, shape, not transpose)
    # ∂data[k] at (row, col) of the original (un-transposed) pattern.
    if transpose:
        row, col = indices[1], indices[0]
    else:
        row, col = indices[0], indices[1]
    row_j = jnp.asarray(row, dtype=jnp.int32)
    col_j = jnp.asarray(col, dtype=jnp.int32)
    contrib = dY[row_j] * X[col_j]
    d_data = contrib.sum(axis=-1) if contrib.ndim == 2 else contrib
    return (d_data, dX)


_spdmm_impl.defvjp(_spdmm_fwd, _spdmm_bwd)


def spdmm(
    A: SparseMatrix,
    X: jax.Array,
    *,
    transpose: bool = False,
    backend: str | None = "auto",
) -> jax.Array:
    """Sparse-dense matmul: returns the dense product ``A @ X`` (or ``Aᵀ @ X``).

    JIT / grad / vmap compatible. Lowers to ``cpu_csr_sparse_dense_ffi`` on
    CPU and ``cusparse_csr_matmat_ffi`` / ``cusparse_csr_matvec_ffi`` on
    GPU; the per-platform dispatch happens at MLIR-lowering time. The
    ``backend`` argument is accepted for symmetry with the SPD ops but
    has no effect — dispatch follows the array's device.
    """
    del backend  # unused; kept for API symmetry
    if X.ndim not in (1, 2):
        raise ValueError(f"spdmm expects X to be 1- or 2-D, got ndim={X.ndim}")
    out_shape = (A.shape[1], A.shape[0]) if transpose else A.shape
    if X.shape[0] != out_shape[1]:
        raise ValueError(f"spdmm shape mismatch: A {out_shape} @ X {X.shape}")
    return _spdmm_impl(A.data, X, A.indices, A.shape, transpose)


# ---------------------------------------------------------------------------
# spmv — same dispatch, simpler shape handling
# ---------------------------------------------------------------------------


def spmv(
    A: SparseMatrix,
    x: jax.Array,
    *,
    transpose: bool = False,
    backend: str | None = "auto",
) -> jax.Array:
    del backend
    out_shape = (A.shape[1], A.shape[0]) if transpose else A.shape
    if x.shape[0] != out_shape[1]:
        raise ValueError(f"spmv shape mismatch: A is {out_shape}, x is {x.shape}")
    return _spdmm_impl(A.data, x, A.indices, A.shape, transpose)
