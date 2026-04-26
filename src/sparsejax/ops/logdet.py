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
    # The ideal solution is still to use the selected inversion algorithm.
    # Current solution:
    # We need A^{-1} only at A's nonzero pattern. Solve A X = E where the
    # columns of E are the standard basis vectors for the unique columns
    # appearing in A.indices[1], then gather A^{-1}[row[k], col[k]] from
    # X[row[k], inv_idx[k]].
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

    return (g * _dispatch_logdet_vjp_data(backend_name, data, indices, shape),)


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
