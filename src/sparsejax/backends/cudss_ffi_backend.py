from __future__ import annotations

from typing import Tuple
import os

import jax
import jax.numpy as jnp
import numpy as np

from sparsejax._csr import coo_to_csr


_REGISTERED = False
_HAVE_HANDLER = False
_HAVE_LOGDET = False
_HAVE_SOLVE_LOGDET = False


def _try_register() -> bool:
    global _REGISTERED, _HAVE_HANDLER, _HAVE_LOGDET, _HAVE_SOLVE_LOGDET
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
        cap_f32 = regs.get("cudss_solve_f32")
        if cap_f32 is not None:
            jax.ffi.register_ffi_target("cudss_solve_f32", cap_f32, platform="CUDA")
    except Exception:
        return False
    _HAVE_HANDLER = True
    ld_cap = regs.get("cudss_logdet")
    if ld_cap is not None:
        try:
            jax.ffi.register_ffi_target("cudss_logdet", ld_cap, platform="CUDA")
            ld_cap_f32 = regs.get("cudss_logdet_f32")
            if ld_cap_f32 is not None:
                jax.ffi.register_ffi_target(
                    "cudss_logdet_f32", ld_cap_f32, platform="CUDA"
                )
            _HAVE_LOGDET = True
        except Exception:
            _HAVE_LOGDET = False
    combo_cap = regs.get("cudss_solve_logdet")
    if combo_cap is not None:
        try:
            jax.ffi.register_ffi_target(
                "cudss_solve_logdet", combo_cap, platform="CUDA"
            )
            combo_cap_f32 = regs.get("cudss_solve_logdet_f32")
            if combo_cap_f32 is not None:
                jax.ffi.register_ffi_target(
                    "cudss_solve_logdet_f32", combo_cap_f32, platform="CUDA"
                )
            _HAVE_SOLVE_LOGDET = True
        except Exception:
            _HAVE_SOLVE_LOGDET = False
    return True


def is_available() -> bool:
    if not _try_register():
        return False
    try:
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False


# matrix_type encoding (must match the C switch in cudss_spd_solve.h)
#   0 -> GENERAL (LU)
#   1 -> SPD     (Cholesky)
#   2 -> SYMMETRIC (LDLᵀ)
_MTYPE = {"general": 0, "spd": 1, "symmetric": 2}
# jax.ffi.ffi_call uses major-to-minor layout notation. For a 2-D (rows, cols) buffer, (1, 0) asks XLA for column-major physical storage.
_COL_MAJOR_2D = (1, 0)


def _dense_ffi_layout(ndim: int) -> tuple[int, int] | None:
    return _COL_MAJOR_2D if ndim == 2 else None


def _env_int(name: str, default: int = -1) -> np.int64:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return np.int64(default)
    return np.int64(int(raw))


def _cudss_config_attrs() -> dict[str, np.int64]:
    """Static cuDSS config knobs read from the environment.

    Supported values are cuDSS enum integer values. For example,
    ``SPARSEJAX_CUDSS_REORDERING_ALG=1`` selects ``CUDSS_ALG_1``.
    ``-1`` leaves the cuDSS default unchanged.
    """

    return {
        "reordering_alg": _env_int("SPARSEJAX_CUDSS_REORDERING_ALG"),
        "factorization_alg": _env_int("SPARSEJAX_CUDSS_FACTORIZATION_ALG"),
        "solve_alg": _env_int("SPARSEJAX_CUDSS_SOLVE_ALG"),
        "use_matching": _env_int("SPARSEJAX_CUDSS_USE_MATCHING"),
        "host_nthreads": _env_int("SPARSEJAX_CUDSS_HOST_NTHREADS"),
        "use_superpanels": _env_int("SPARSEJAX_CUDSS_USE_SUPERPANELS"),
    }


def _cudss_target(base: str, dtype) -> str:
    dt = jnp.dtype(dtype)
    if dt == jnp.float32:
        return f"{base}_f32"
    if dt == jnp.float64:
        return base
    raise TypeError(f"cudss_ffi supports float32/float64, got {dt}")


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
    upper = matrix_type in ("spd", "symmetric")
    # Current native handlers bind ffi::Buffer<S32> and CUDA_R_32I.
    csr = coo_to_csr(indices, shape, index_dtype=np.int32, upper=upper)
    # Permute the traced data onto CSR layout in-graph.
    data_csr = data if csr.order_is_identity else data[csr.order]
    # Broadcast static int arrays onto the right device as jnp arrays.
    row_ptr = jnp.asarray(csr.indptr)
    col_idx = jnp.asarray(csr.col_idx)

    out_dtype = jnp.result_type(data.dtype, b.dtype)
    if out_dtype not in (jnp.float32, jnp.float64):
        out_dtype = jnp.float64
    data_csr = data_csr.astype(out_dtype)
    b_typed = b.astype(out_dtype)
    dense_layout = _dense_ffi_layout(b.ndim)

    ffi_fn = jax.ffi.ffi_call(
        _cudss_target("cudss_solve", out_dtype),
        jax.ShapeDtypeStruct(b.shape, out_dtype),
        input_layouts=(None, None, None, dense_layout),
        output_layouts=dense_layout,
        vmap_method="sequential",
    )
    config_attrs = _cudss_config_attrs()
    mt = _MTYPE[matrix_type]
    x = ffi_fn(
        row_ptr,
        col_idx,
        data_csr,
        b_typed,
        matrix_type=np.int64(mt),
        factor_token=np.int64(csr.factor_token),
        **config_attrs,
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


def cholesky_solve_and_logdet(
    data: jax.Array,
    indices: np.ndarray,
    shape: Tuple[int, int],
    b: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    if not _try_register() or not _HAVE_SOLVE_LOGDET:
        x = cholesky_solve(data, indices, shape, b)
        ld = logdet(data, indices, shape, matrix_type="spd")
        return x, ld

    m, n = shape
    if m != n:
        raise ValueError(f"cuDSS requires square matrix, got {shape}")
    # Current native handlers bind ffi::Buffer<S32> and CUDA_R_32I.
    csr = coo_to_csr(indices, shape, index_dtype=np.int32, upper=True)
    data_csr = data if csr.order_is_identity else data[csr.order]
    out_dtype = jnp.result_type(data.dtype, b.dtype)
    if out_dtype not in (jnp.float32, jnp.float64):
        out_dtype = jnp.float64
    data_csr = data_csr.astype(out_dtype)
    row_ptr = jnp.asarray(csr.indptr)
    col_idx = jnp.asarray(csr.col_idx)
    b_typed = b.astype(out_dtype)
    dense_layout = _dense_ffi_layout(b.ndim)

    ffi_fn = jax.ffi.ffi_call(
        _cudss_target("cudss_solve_logdet", out_dtype),
        (
            jax.ShapeDtypeStruct(b.shape, out_dtype),
            jax.ShapeDtypeStruct((), out_dtype),
        ),
        input_layouts=(None, None, None, dense_layout),
        output_layouts=(dense_layout, None),
        vmap_method="sequential",
    )
    config_attrs = _cudss_config_attrs()
    mt = _MTYPE["spd"]
    x, ld = ffi_fn(
        row_ptr,
        col_idx,
        data_csr,
        b_typed,
        matrix_type=np.int64(mt),
        factor_token=np.int64(csr.factor_token),
        **config_attrs,
    )
    return x.astype(jnp.result_type(data.dtype, b.dtype)), ld.astype(data.dtype)


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

    upper = matrix_type in ("spd", "symmetric")
    # Current native handlers bind ffi::Buffer<S32> and CUDA_R_32I.
    csr = coo_to_csr(indices, shape, index_dtype=np.int32, upper=upper)
    data_csr = data if csr.order_is_identity else data[csr.order]
    out_dtype = data.dtype if data.dtype in (jnp.float32, jnp.float64) else jnp.float64
    data_csr = data_csr.astype(out_dtype)
    row_ptr = jnp.asarray(csr.indptr)
    col_idx = jnp.asarray(csr.col_idx)

    ffi_fn = jax.ffi.ffi_call(
        _cudss_target("cudss_logdet", out_dtype),
        jax.ShapeDtypeStruct((), out_dtype),
        vmap_method="sequential",
    )
    config_attrs = _cudss_config_attrs()
    mt = _MTYPE[matrix_type]
    ld = ffi_fn(
        row_ptr,
        col_idx,
        data_csr,
        matrix_type=np.int64(mt),
        factor_token=np.int64(csr.factor_token),
        **config_attrs,
    )
    return ld.astype(data.dtype)
