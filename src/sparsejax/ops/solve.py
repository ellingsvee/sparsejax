from __future__ import annotations

from functools import partial

import jax
import numpy as np

from sparsejax.sparse import SparseMatrix
from sparsejax.utils import _resolve_backend


def _dispatch_solve(
    backend_name: str,
    data: jax.Array,
    row: np.ndarray,
    col: np.ndarray,
    shape: tuple,
    b: jax.Array,
    method: str,
) -> jax.Array:
    if backend_name == "scipy":
        from sparsejax.backends import scipy_backend

        return scipy_backend.solve(data, row, col, shape, b, method=method)
    if backend_name == "cholmod":
        from sparsejax.backends import cholmod_backend

        indices = np.stack([np.asarray(row), np.asarray(col)], axis=0)
        return cholmod_backend.cholesky_solve(data, indices, shape, b)
    if backend_name == "cudss_ffi":
        from sparsejax.backends import cudss_ffi_backend

        indices = np.stack([np.asarray(row), np.asarray(col)], axis=0)
        return cudss_ffi_backend.solve(data, indices, shape, b, method=method)
    if backend_name == "cudss":
        from sparsejax.backends import cudss_backend

        return cudss_backend.solve(data, row, col, shape, b, method=method)
    raise ValueError(f"unknown backend: {backend_name!r}")


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5, 6))
def _spsolve_impl(
    data: jax.Array,
    b: jax.Array,
    indices: np.ndarray,
    shape: tuple,
    backend_name: str,
    method: str,
    transpose: bool,
):
    if transpose:
        row, col = indices[1], indices[0]
        shp = (shape[1], shape[0])
    else:
        row, col = indices[0], indices[1]
        shp = shape
    return _dispatch_solve(backend_name, data, row, col, shp, b, method)


def _spsolve_fwd(data, b, indices, shape, backend_name, method, transpose):
    x = _spsolve_impl(data, b, indices, shape, backend_name, method, transpose)
    return x, (data, x)


def _spsolve_bwd(indices, shape, backend_name, method, transpose, residuals, g):
    data, x = residuals
    lam = _spsolve_impl(data, g, indices, shape, backend_name, method, not transpose)
    if transpose:
        row_idx, col_idx = indices[1], indices[0]
    else:
        row_idx, col_idx = indices[0], indices[1]
    g_data = -lam[row_idx] * x[col_idx]
    if g_data.ndim == 2:
        g_data = g_data.sum(axis=-1)
    return (g_data, lam)


_spsolve_impl.defvjp(_spsolve_fwd, _spsolve_bwd)


def spsolve(
    A: SparseMatrix,
    b: jax.Array,
    *,
    backend: str | None = "auto",
    method: str = "lu",
    spd: bool = False,
    transpose: bool = False,
) -> jax.Array:
    if b.shape[0] != A.shape[0]:
        raise ValueError(f"shape mismatch: A is {A.shape}, b is {b.shape}")
    backend_name = _resolve_backend(A, backend, spd=spd)
    return _spsolve_impl(A.data, b, A.indices, A.shape, backend_name, method, transpose)
