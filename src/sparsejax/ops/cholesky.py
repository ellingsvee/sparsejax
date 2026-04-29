from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from sparsejax.dense_mode import is_dense_mode
from sparsejax.sparse import SparseMatrix
from sparsejax.utils import _resolve_backend


def _dense_cholesky_solve(A: SparseMatrix, b: jax.Array) -> jax.Array:
    Ad = A.to_dense()
    L = jnp.linalg.cholesky(Ad)
    return jax.scipy.linalg.cho_solve((L, True), b)


def _dense_cholesky_solve_and_logdet(
    A: SparseMatrix, b: jax.Array
) -> tuple[jax.Array, jax.Array]:
    Ad = A.to_dense()
    L = jnp.linalg.cholesky(Ad)
    x = jax.scipy.linalg.cho_solve((L, True), b)
    ld = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    return x, ld


def _dispatch_chol_solve(
    backend_name: str,
    data: jax.Array,
    indices: np.ndarray,
    shape: tuple,
    b: jax.Array,
) -> jax.Array:
    if backend_name in ("cholmod", "cholmod_takahashi"):
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


def _dispatch_chol_solve_and_logdet(
    backend_name: str,
    data: jax.Array,
    indices: np.ndarray,
    shape: tuple,
    b: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    if backend_name in ("cholmod", "cholmod_takahashi"):
        from sparsejax.backends import cholmod_backend

        return cholmod_backend.cholesky_solve_and_logdet(data, indices, shape, b)
    if backend_name == "scipy":
        from sparsejax.backends import scipy_backend

        return scipy_backend.solve_and_logdet(data, indices, shape, b)
    if backend_name == "cudss_ffi":
        from sparsejax.backends import cudss_ffi_backend

        return cudss_ffi_backend.cholesky_solve_and_logdet(data, indices, shape, b)

    x = _dispatch_chol_solve(backend_name, data, indices, shape, b)
    from .logdet import _dispatch_logdet

    ld = _dispatch_logdet(backend_name, data, indices, shape)
    return x, ld


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
    g_data = _sym_upper_solve_data_grad(indices, lam, x)
    return (g_data, lam)


_cholesky_solve_impl.defvjp(_cholesky_solve_fwd, _cholesky_solve_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4))
def _cholesky_solve_and_logdet_impl(
    data: jax.Array,
    b: jax.Array,
    indices: np.ndarray,
    shape: tuple,
    backend_name: str,
):
    return _dispatch_chol_solve_and_logdet(backend_name, data, indices, shape, b)


def _cholesky_solve_and_logdet_fwd(data, b, indices, shape, backend_name):
    x, ld = _dispatch_chol_solve_and_logdet(backend_name, data, indices, shape, b)
    return (x, ld), (data, x)


def _cholesky_solve_and_logdet_bwd(indices, shape, backend_name, residuals, g):
    data, x = residuals
    gx, gld = g
    row_idx = indices[0]
    col_idx = indices[1]
    row_arr = np.asarray(row_idx, dtype=np.int64)
    col_arr = np.asarray(col_idx, dtype=np.int64)

    if backend_name == "cholmod_takahashi":
        from sparsejax.backends import cholmod_backend

        lam = _dispatch_chol_solve(backend_name, data, indices, shape, gx)
        logdet_inv_data = cholmod_backend.selected_inverse_entries_takahashi(
            data, indices, shape
        )
    else:
        n = shape[0]
        unique_cols, inv_idx = np.unique(col_arr, return_inverse=True)
        E = jax.nn.one_hot(unique_cols, n, dtype=data.dtype).T

        if backend_name in ("cholmod", "scipy"):
            if gx.ndim == 1:
                gx_cols = 1
                gx_rhs = gx[:, None]
            else:
                gx_cols = gx.shape[1]
                gx_rhs = gx
            rhs = jnp.concatenate([gx_rhs, E], axis=1)
            sol = _dispatch_chol_solve(backend_name, data, indices, shape, rhs)
            lam_mat = sol[:, :gx_cols]
            inv_cols = sol[:, gx_cols:]
            lam = lam_mat[:, 0] if gx.ndim == 1 else lam_mat
        else:
            lam = _dispatch_chol_solve(backend_name, data, indices, shape, gx)
            inv_cols = _dispatch_chol_solve(backend_name, data, indices, shape, E)
        logdet_inv_data = inv_cols[jnp.asarray(row_arr), jnp.asarray(inv_idx)]

    solve_g_data = _sym_upper_solve_data_grad(indices, lam, x)
    offdiag = row_idx != col_idx
    upper = row_idx <= col_idx
    logdet_inv_data = jnp.where(offdiag, 2.0 * logdet_inv_data, logdet_inv_data)
    logdet_inv_data = jnp.where(upper, logdet_inv_data, jnp.zeros_like(logdet_inv_data))
    logdet_g_data = gld * logdet_inv_data
    return (solve_g_data + logdet_g_data, lam)


_cholesky_solve_and_logdet_impl.defvjp(
    _cholesky_solve_and_logdet_fwd, _cholesky_solve_and_logdet_bwd
)


def _sym_upper_solve_data_grad(indices: np.ndarray, lam: jax.Array, x: jax.Array):
    """Gradient for an SPD matrix represented by its active upper triangle."""
    row_idx = indices[0]
    col_idx = indices[1]
    contrib = -lam[row_idx] * x[col_idx]
    offdiag = row_idx != col_idx
    contrib_t = -lam[col_idx] * x[row_idx]
    contrib = jnp.where(
        offdiag[..., None] if contrib.ndim == 2 else offdiag,
        contrib + contrib_t,
        contrib,
    )
    if contrib.ndim == 2:
        contrib = contrib.sum(axis=-1)
    upper = row_idx <= col_idx
    return jnp.where(upper, contrib, jnp.zeros_like(contrib))


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
    if is_dense_mode():
        return _dense_cholesky_solve(A, b)
    backend_name = _resolve_backend(A, backend)
    return _cholesky_solve_impl(A.data, b, A.indices, A.shape, backend_name)


def cholesky_solve_and_logdet(
    A: SparseMatrix,
    b: jax.Array,
    *,
    backend: str | None = "auto",
) -> tuple[jax.Array, jax.Array]:
    """Solve ``A x = b`` and return ``(x, logdet(A))`` from one factorization.

    On the CHOLMOD backend this is a single host callback and a single numeric
    factorization in the forward pass. Its VJP batches the solve cotangent and
    logdet inverse-column solve into one backend solve.
    """
    if b.shape[0] != A.shape[0]:
        raise ValueError(f"shape mismatch: A is {A.shape}, b is {b.shape}")
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"logdet requires square matrix, got {A.shape}")
    if is_dense_mode():
        return _dense_cholesky_solve_and_logdet(A, b)
    backend_name = _resolve_backend(A, backend)
    return _cholesky_solve_and_logdet_impl(A.data, b, A.indices, A.shape, backend_name)


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

    Returns an object you can call like ``factor(b)``. Not usable inside jit. This is a Python-level convenience for the factor-once, solve-many pattern.
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
