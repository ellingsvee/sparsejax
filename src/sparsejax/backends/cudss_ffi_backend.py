from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from sparsejax._csr import coo_to_csr


_REGISTERED = False
_HAVE_HANDLER = False
_HAVE_LOGDET = False


def _try_register() -> bool:
    global _REGISTERED, _HAVE_HANDLER, _HAVE_LOGDET
    if _REGISTERED:
        return _HAVE_HANDLER
    _REGISTERED = True
    try:
        from sparsejax import backend as _native  # type: ignore
    except ImportError:
        return False
    try:
        regs = _native.registrations()
    except Exception:
        return False
    cap = regs.get("cudss_solve")
    if cap is None:
        return False
    try:
        jax.ffi.register_ffi_target("cudss_solve", cap, platform="CUDA")
    except Exception:
        return False
    _HAVE_HANDLER = True
    ld_cap = regs.get("cudss_logdet")
    if ld_cap is not None:
        try:
            jax.ffi.register_ffi_target("cudss_logdet", ld_cap, platform="CUDA")
            _HAVE_LOGDET = True
        except Exception:
            _HAVE_LOGDET = False
    return True


def is_available() -> bool:
    if not _try_register():
        return False
    try:
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False


# --------------------------------------------------------------------------
# matrix_type encoding (must match the C switch in cudss_spd_solve.h)
#   0 -> GENERAL (LU)
#   1 -> SPD     (Cholesky)
#   2 -> SYMMETRIC (LDLᵀ)
# --------------------------------------------------------------------------
_MTYPE = {"general": 0, "spd": 1, "symmetric": 2}


def _call_cudss(
    data: jax.Array,
    indices: np.ndarray,
    shape: Tuple[int, int],
    b: jax.Array,
    matrix_type: str,
) -> jax.Array:
    if not _try_register():
        raise RuntimeError("cudss_ffi backend: native handler not available")
    m, n = shape
    if m != n:
        raise ValueError(f"cuDSS requires square matrix, got {shape}")
    csr = coo_to_csr(indices, shape)
    # Permute the traced data onto CSR layout in-graph.
    data_csr = data[csr.order]
    # Broadcast static int arrays onto the right device as jnp arrays.
    row_ptr = jnp.asarray(csr.indptr, dtype=jnp.int32)
    col_idx = jnp.asarray(csr.col_idx, dtype=jnp.int32)

    # cuDSS handler is F64 only; cast b/data if needed.
    data_csr = data_csr.astype(jnp.float64)
    b64 = b.astype(jnp.float64)

    out_dtype = jnp.float64
    out_shape = b.shape

    ffi_fn = jax.ffi.ffi_call(
        "cudss_solve",
        jax.ShapeDtypeStruct(out_shape, out_dtype),
        vmap_method="sequential",
    )
    mt = _MTYPE[matrix_type]
    x = ffi_fn(
        row_ptr,
        col_idx,
        data_csr,
        b64,
        matrix_type=np.int64(mt),
        factor_token=np.int64(csr.factor_token),
    )
    return x.astype(jnp.result_type(data.dtype, b.dtype))


def solve(
    data: jax.Array,
    indices: np.ndarray,
    shape: Tuple[int, int],
    b: jax.Array,
    *,
    method: str = "lu",
) -> jax.Array:
    if method != "lu":
        raise ValueError(f"cudss_ffi solve supports method='lu', got {method!r}")
    return _call_cudss(data, indices, shape, b, "general")


def cholesky_solve(
    data: jax.Array,
    indices: np.ndarray,
    shape: Tuple[int, int],
    b: jax.Array,
) -> jax.Array:
    return _call_cudss(data, indices, shape, b, "spd")


def build_factor(
    data: jax.Array,
    indices: np.ndarray,
    shape: Tuple[int, int],
):
    from sparsejax.ops.cholesky import CholeskyFactor
    from sparsejax.sparse import SparseMatrix

    A = SparseMatrix(data=data, indices=indices, shape=shape)

    def _solve(b: jax.Array) -> jax.Array:
        return cholesky_solve(A.data, indices, shape, b)

    ld = float(jnp.asarray(logdet(A.data, indices, shape, matrix_type="spd")))
    return CholeskyFactor(solve_fn=_solve, logdet_val=ld)


def logdet(
    data: jax.Array,
    indices: np.ndarray,
    shape: Tuple[int, int],
    *,
    matrix_type: str = "spd",
) -> jax.Array:
    if not _try_register() or not _HAVE_LOGDET:
        from .cholmod_backend import logdet as cpu_logdet

        return cpu_logdet(data, indices, shape)

    m, n = shape
    if m != n:
        raise ValueError(f"logdet requires square matrix, got {shape}")

    from sparsejax._csr import coo_to_csr

    csr = coo_to_csr(indices, shape)
    data_csr = data[csr.order].astype(jnp.float64)
    row_ptr = jnp.asarray(csr.indptr, dtype=jnp.int32)
    col_idx = jnp.asarray(csr.col_idx, dtype=jnp.int32)

    out_dtype = jnp.float64
    ffi_fn = jax.ffi.ffi_call(
        "cudss_logdet",
        jax.ShapeDtypeStruct((), out_dtype),
        vmap_method="sequential",
    )
    mt = _MTYPE[matrix_type]
    ld = ffi_fn(
        row_ptr,
        col_idx,
        data_csr,
        matrix_type=np.int64(mt),
        factor_token=np.int64(csr.factor_token),
    )
    return ld.astype(data.dtype)
