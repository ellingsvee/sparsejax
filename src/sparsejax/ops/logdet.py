from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from sparsejax.sparse import SparseMatrix

from sparsejax.utils import _resolve_backend


def _dispatch_logdet(
    backend_name: str, data: jax.Array, indices: np.ndarray, shape: tuple
) -> jax.Array:
    if backend_name == "cholmod":
        from sparsejax.backends import cholmod_backend

        return cholmod_backend.logdet(data, indices, shape)
    if backend_name == "cudss_ffi":
        from sparsejax.backends import cudss_ffi_backend

        return cudss_ffi_backend.logdet(data, indices, shape)
    if backend_name == "cudss":
        from sparsejax.backends import cudss_backend

        return cudss_backend.logdet(data, indices, shape)
    if backend_name == "scipy":
        from sparsejax.backends import scipy_backend

        return scipy_backend.logdet(data, indices, shape)
    raise ValueError(f"unknown logdet backend: {backend_name!r}")


def _dispatch_chol_solve(
    backend_name: str,
    data: jax.Array,
    indices: np.ndarray,
    shape: tuple,
    b: jax.Array,
) -> jax.Array:
    from .cholesky import _dispatch_chol_solve as _d

    return _d(backend_name, data, indices, shape, b)


@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3))
def _logdet_impl(
    data: jax.Array,
    indices: np.ndarray,
    shape: tuple,
    backend_name: str,
):
    return _dispatch_logdet(backend_name, data, indices, shape)


def _logdet_fwd(data, indices, shape, backend_name):
    val = _dispatch_logdet(backend_name, data, indices, shape)
    return val, (data,)


def _logdet_bwd(indices, shape, backend_name, residuals, g):
    (data,) = residuals

    # TODO: Here we need to implement the partial inverse. The current approach for solving against identity is suboptimal.
    # Solve A X = I (restricted to unique column set) to get selected
    # entries of A^{-1}. For SPD, A^{-1} is symmetric; we need A^{-1}[row,col]
    # at every entry of the pattern.
    col_arr = np.asarray(indices[1], dtype=np.int64)
    row_arr = np.asarray(indices[0], dtype=np.int64)
    n = shape[0]
    I = jnp.eye(n, dtype=data.dtype)
    A_inv = _dispatch_chol_solve(backend_name, data, indices, shape, I)
    g_data = g * A_inv[row_arr, col_arr]
    return (g_data,)


_logdet_impl.defvjp(_logdet_fwd, _logdet_bwd)


def logdet(
    A: SparseMatrix,
    *,
    backend: str | None = "auto",
) -> jax.Array:
    """Log-determinant of an SPD sparse matrix."""
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"logdet requires square matrix, got {A.shape}")
    backend_name = _resolve_backend(A, backend)
    return _logdet_impl(A.data, A.indices, A.shape, backend_name)
