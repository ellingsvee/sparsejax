from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from sparsejax.sparse import SparseMatrix
from sparsejax.utils import _resolve_backend


def _dispatch_chol_solve(
    backend_name: str,
    data: jax.Array,
    indices: np.ndarray,
    shape: tuple,
    b: jax.Array,
) -> jax.Array:
    if backend_name == "cholmod":
        from sparsejax.backends import cholmod_backend

        return cholmod_backend.cholesky_solve(data, indices, shape, b)
    if backend_name == "cudss_ffi":
        from sparsejax.backends import cudss_ffi_backend

        return cudss_ffi_backend.cholesky_solve(data, indices, shape, b)
    if backend_name == "cudss":
        from sparsejax.backends import cudss_backend

        return cudss_backend.cholesky_solve(data, indices, shape, b)
    if backend_name == "scipy":
        # scipy has no Cholesky; fall back to LU which still produces the
        # same solve for SPD matrices (slower).
        from sparsejax.backends import scipy_backend

        return scipy_backend.solve(data, indices[0], indices[1], shape, b, method="lu")
    raise ValueError(f"unknown cholesky backend: {backend_name!r}")


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4))
def _cholesky_solve_impl(
    data: jax.Array,
    b: jax.Array,
    indices: np.ndarray,
    shape: tuple,
    backend_name: str,
):
    return _dispatch_chol_solve(backend_name, data, indices, shape, b)


def _cholesky_solve_fwd(data, b, indices, shape, backend_name):
    x = _dispatch_chol_solve(backend_name, data, indices, shape, b)
    return x, (data, x)


def _cholesky_solve_bwd(indices, shape, backend_name, residuals, g):
    data, x = residuals
    # For SPD: A^{-T} = A^{-1}, so ∂b: λ = A^{-1} g = cholesky_solve(A, g)
    lam = _dispatch_chol_solve(backend_name, data, indices, shape, g)
    # ∂A/∂data_k at (row_k, col_k): -λ[row] * x[col]
    row_idx = indices[0]
    col_idx = indices[1]
    g_data = -lam[row_idx] * x[col_idx]
    if g_data.ndim == 2:
        g_data = g_data.sum(axis=-1)
    return (g_data, lam)


_cholesky_solve_impl.defvjp(_cholesky_solve_fwd, _cholesky_solve_bwd)


def cholesky_solve(
    A: SparseMatrix,
    b: jax.Array,
    *,
    backend: str | None = "auto",
) -> jax.Array:
    """Solve ``A x = b`` for a sparse SPD ``A``.

    Uses CHOLMOD on CPU and cuDSS (via FFI if available) on GPU. AD is
    supported through ``data`` and ``b``.
    """
    if b.shape[0] != A.shape[0]:
        raise ValueError(f"shape mismatch: A is {A.shape}, b is {b.shape}")
    backend_name = _resolve_backend(A, backend)
    return _cholesky_solve_impl(A.data, b, A.indices, A.shape, backend_name)


class CholeskyFactor:
    """An opaque factor object. Call it on a vector to solve."""

    def __init__(self, solve_fn, logdet_val: float | None):
        self._solve = solve_fn
        self._logdet = logdet_val

    def __call__(self, b: jax.Array) -> jax.Array:
        return self._solve(b)

    def solve(self, b: jax.Array) -> jax.Array:
        return self._solve(b)

    @property
    def logdet(self) -> float:
        if self._logdet is None:
            raise RuntimeError("logdet not computed by this backend")
        return self._logdet


def cholesky_factor(
    A: SparseMatrix,
    *,
    backend: str | None = "auto",
) -> CholeskyFactor:
    """Factor ``A`` (SPD) for repeated solves.

    Returns an object you can call like ``factor(b)``. Not usable inside
    jit — this is a Python-level convenience for the factor-once,
    solve-many pattern.
    """
    backend_name = _resolve_backend(A, backend)
    data = np.asarray(A.data)
    if backend_name == "cholmod":
        from sparsejax.backends import cholmod_backend

        return cholmod_backend.build_factor(data, A.indices, A.shape)
    if backend_name == "cudss_ffi":
        from sparsejax.backends import cudss_ffi_backend

        return cudss_ffi_backend.build_factor(A.data, A.indices, A.shape)
    if backend_name == "cudss":
        from sparsejax.backends import cudss_backend

        return cudss_backend.build_factor(data, A.indices, A.shape)
    if backend_name == "scipy":
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        mat = sp.coo_matrix((data, (A.indices[0], A.indices[1])), shape=A.shape).tocsc()
        lu = spla.splu(mat)

        def solve(b):
            return jnp.asarray(lu.solve(np.asarray(b)))

        return CholeskyFactor(solve_fn=solve, logdet_val=None)
    raise ValueError(f"unknown backend: {backend_name!r}")
