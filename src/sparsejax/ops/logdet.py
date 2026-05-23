from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from sparsejax.sparse import SparseMatrix

from sparsejax.dense_mode import is_dense_mode
from sparsejax.utils import _resolve_backend


def _dispatch_logdet(
    backend_name: str, data: jax.Array, indices: np.ndarray, shape: tuple
) -> jax.Array:
    if backend_name in ("cholmod", "cholmod_takahashi"):
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


def _dispatch_logdet_vjp_data(
    backend_name: str,
    data: jax.Array,
    indices: np.ndarray,
    shape: tuple,
) -> jax.Array:
    if backend_name == "cholmod_takahashi":
        from sparsejax.backends import cholmod_backend

        return cholmod_backend.selected_inverse_entries_takahashi(data, indices, shape)

    # TODO: There is still some redundant work here when A has multiple entries in the same column.
    # The ideal solution is still to use a GPU-version of the selected inversion algorithm.
    # However, this is hard, as there is some serial work needed for the selected inversion.
    row_arr = np.asarray(indices[0], dtype=np.int64)
    col_arr = np.asarray(indices[1], dtype=np.int64)
    n = shape[0]
    unique_cols, inv_idx = np.unique(col_arr, return_inverse=True)
    E = jax.nn.one_hot(unique_cols, n, dtype=data.dtype).T
    X = _dispatch_chol_solve(backend_name, data, indices, shape, E)
    return X[jnp.asarray(row_arr), jnp.asarray(inv_idx)]


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
    inv_data = _dispatch_logdet_vjp_data(backend_name, data, indices, shape)
    row_idx = indices[0]
    col_idx = indices[1]
    offdiag = row_idx != col_idx
    upper = row_idx <= col_idx
    inv_data = jnp.where(offdiag, 2.0 * inv_data, inv_data)
    inv_data = jnp.where(upper, inv_data, jnp.zeros_like(inv_data))

    return (g * inv_data,)


_logdet_impl.defvjp(_logdet_fwd, _logdet_bwd)


def logdet(
    A: SparseMatrix,
    *,
    backend: str | None = "auto",
) -> jax.Array:
    """Log-determinant of an SPD sparse matrix."""
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"logdet requires square matrix, got {A.shape}")
    if is_dense_mode():
        return jnp.linalg.slogdet(A.to_dense())[1]
    backend_name = _resolve_backend(A, backend)
    return _logdet_impl(A.data, A.indices, A.shape, backend_name)
