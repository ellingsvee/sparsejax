"""cuDSS/cuSPARSE GPU backend via ``nvmath-python`` + ``cupy``. Prefer :mod:`cudss_ffi_backend` whenever possible..."""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np


def _require_nvmath():
    try:
        import cupy  # noqa: F401
        import cupyx.scipy.sparse  # noqa: F401
        from nvmath.sparse.advanced import direct_solver, DirectSolverMatrixType
    except ImportError as e:  # pragma: no cover - env-dependent
        raise RuntimeError(
            "cudss backend requires nvmath-python[cu13] and cupy."
        ) from e
    return direct_solver, DirectSolverMatrixType


_MATRIX_TYPES = {"general": "GENERAL", "spd": "SPD", "symmetric": "SYMMETRIC"}


def _cudss_solve_host(
    data_h,
    b_h,
    row,
    col,
    shape: Tuple[int, int],
    matrix_type: str,
):
    direct_solver, DirectSolverMatrixType = _require_nvmath()
    import cupy as cp
    import cupyx.scipy.sparse as cusp

    m, n = shape
    if m != n:
        raise ValueError(f"cuDSS requires square matrix, got {shape}")

    data_gpu = cp.asarray(data_h)
    row_gpu = cp.asarray(row, dtype=cp.int32)
    col_gpu = cp.asarray(col, dtype=cp.int32)
    A = cusp.coo_matrix((data_gpu, (row_gpu, col_gpu)), shape=shape).tocsr()
    # nvmath requires col-major layout for multi-column RHS.
    b_np = np.asarray(b_h)
    if b_np.ndim == 2:
        b_np = np.asfortranarray(b_np)
    b_gpu = cp.asarray(b_np)

    mtype_name = _MATRIX_TYPES.get(matrix_type)
    if mtype_name is None:
        raise ValueError(
            f"unknown cuDSS matrix_type {matrix_type!r}; "
            f"expected one of {list(_MATRIX_TYPES)}"
        )
    options = {"sparse_system_type": getattr(DirectSolverMatrixType, mtype_name)}
    x_gpu = direct_solver(A, b_gpu, options=options)
    if not isinstance(x_gpu, cp.ndarray):
        x_gpu = cp.asarray(x_gpu)
    return cp.asnumpy(x_gpu).astype(data_h.dtype, copy=False)


def _solve_common(data, row, col, shape, b, matrix_type: str):
    out_shape = b.shape
    out_dtype = jnp.result_type(data.dtype, b.dtype)
    row_s = np.asarray(row, dtype=np.int32)
    col_s = np.asarray(col, dtype=np.int32)
    shape_s = tuple(shape)

    def _host(data_h, b_h):
        return _cudss_solve_host(
            np.asarray(data_h),
            np.asarray(b_h),
            row_s,
            col_s,
            shape_s,
            matrix_type,
        )

    return jax.pure_callback(
        _host,
        jax.ShapeDtypeStruct(out_shape, out_dtype),
        data,
        b,
    )


def solve(
    data,
    row,
    col,
    shape: Tuple[int, int],
    b,
    *,
    method: str = "lu",
):
    if method != "lu":
        raise ValueError(f"cuDSS backend only supports method='lu', got {method!r}")
    return _solve_common(data, row, col, shape, b, "general")


def cholesky_solve(
    data: jax.Array,
    indices: np.ndarray,
    shape: Tuple[int, int],
    b: jax.Array,
) -> jax.Array:
    return _solve_common(data, indices[0], indices[1], shape, b, "spd")


def logdet(
    data: jax.Array,
    indices: np.ndarray,
    shape: Tuple[int, int],
) -> jax.Array:
    # Delegate to CHOLMOD — we already need to pull values to host for a
    # factor; there is no cheap GPU path through nvmath for log-det.
    from .cholmod_backend import logdet as cpu_logdet

    return cpu_logdet(data, indices, shape)


def build_factor(data: np.ndarray, indices: np.ndarray, shape):
    """Factor-once API using nvmath's direct_solver.

    ``nvmath.sparse.advanced.direct_solver`` does not yet expose a
    factor-reuse handle; we cache the CSR buffers on device and build a
    fresh factor per call. The host-side callback model pays for a
    cupy memcopy per solve. For the fast path use CHOLMOD (CPU) or the
    FFI handler (see :mod:`cudss_ffi_backend`).
    """
    from sparsejax.ops.cholesky import CholeskyFactor
    import cupy as cp
    import cupyx.scipy.sparse as cusp

    direct_solver, DirectSolverMatrixType = _require_nvmath()

    data_np = np.asarray(data)
    data_gpu = cp.asarray(data_np)
    row_gpu = cp.asarray(indices[0], dtype=cp.int32)
    col_gpu = cp.asarray(indices[1], dtype=cp.int32)
    A_gpu = cusp.coo_matrix((data_gpu, (row_gpu, col_gpu)), shape=shape).tocsr()
    options = {"sparse_system_type": DirectSolverMatrixType.SPD}

    def _solve(b):
        b_np = np.asarray(b)
        b_gpu = cp.asarray(b_np)
        x_gpu = direct_solver(A_gpu, b_gpu, options=options)
        if not isinstance(x_gpu, cp.ndarray):
            x_gpu = cp.asarray(x_gpu)
        return jnp.asarray(cp.asnumpy(x_gpu))

    # logdet via CPU factor for simplicity
    from .cholmod_backend import build_factor as cpu_build_factor

    try:
        cpu_factor = cpu_build_factor(data_np, indices, shape)
        ld = cpu_factor.logdet
    except Exception:
        ld = None
    return CholeskyFactor(solve_fn=_solve, logdet_val=ld)


def spspmm(
    a_data,
    a_row,
    a_col,
    a_shape: Tuple[int, int],
    b_data,
    b_row,
    b_col,
    b_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    """SpGEMM via cupy (cuSPARSE). Shape-dependent output, so not
    jit-compatible; called directly from the ops layer."""
    try:
        import cupy as cp
        import cupyx.scipy.sparse as cusp
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("cuDSS spspmm requires cupy.") from e

    A = cusp.coo_matrix(
        (
            cp.asarray(np.asarray(a_data)),
            (cp.asarray(a_row, dtype=cp.int32), cp.asarray(a_col, dtype=cp.int32)),
        ),
        shape=a_shape,
    ).tocsr()
    B = cusp.coo_matrix(
        (
            cp.asarray(np.asarray(b_data)),
            (cp.asarray(b_row, dtype=cp.int32), cp.asarray(b_col, dtype=cp.int32)),
        ),
        shape=b_shape,
    ).tocsr()
    C = (A @ B).tocoo()
    return (
        cp.asnumpy(C.data),
        cp.asnumpy(C.row).astype(np.int32, copy=False),
        cp.asnumpy(C.col).astype(np.int32, copy=False),
        (a_shape[0], b_shape[1]),
    )
