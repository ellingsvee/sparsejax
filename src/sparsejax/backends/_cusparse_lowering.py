"""
Lowerings adapted from the jax.experimental.sparse module
"""

from __future__ import annotations

from functools import lru_cache, partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import core
from jax._src import dispatch
from jax._src import ffi as _jax_ffi
from jax._src.interpreters import mlir


_REGISTERED = False
_HAVE_GPU = False
_HAVE_CPU = False


def _ensure_registered() -> None:
    """Forward jaxlib's sparse FFI capsules into the JAX FFI registry."""
    global _REGISTERED, _HAVE_GPU, _HAVE_CPU
    if _REGISTERED:
        return
    try:
        from jaxlib import gpu_sparse  # type: ignore

        for platform, targets in gpu_sparse.registrations().items():
            for name, capsule, api_version in targets:
                try:
                    jax.ffi.register_ffi_target(
                        name,
                        capsule,
                        platform=platform,
                        api_version=api_version,
                    )
                except Exception:
                    pass
            if targets:
                _HAVE_GPU = True
    except ImportError:  # pragma: no cover
        pass

    try:
        from jaxlib import cpu_sparse  # type: ignore

        for platform, targets in cpu_sparse.registrations().items():
            for name, capsule, api_version in targets:
                try:
                    jax.ffi.register_ffi_target(
                        name,
                        capsule,
                        platform=platform,
                        api_version=api_version,
                    )
                except Exception:
                    pass
            if targets:
                _HAVE_CPU = True
    except ImportError:  # pragma: no cover
        pass
    _REGISTERED = True


# Make sure FFI targets are visible before any lowering compiles.
_ensure_registered()


@lru_cache(maxsize=128)
def _csr_matvec_descriptor(
    rows: int, cols: int, nnz: int, transpose: bool, dtype_str: str
) -> Tuple[int, bytes]:
    from jaxlib import gpu_sparse  # type: ignore

    dt = np.dtype(dtype_str)
    buf_size, opaque = gpu_sparse._cusparse.build_csr_matvec_descriptor(
        dt, dt, dt, np.dtype(np.int32), rows, cols, nnz, transpose
    )
    return int(buf_size), bytes(opaque)


@lru_cache(maxsize=128)
def _csr_matmat_descriptor(
    rows: int, cols: int, ncols: int, nnz: int, transpose: bool, dtype_str: str
) -> Tuple[int, bytes]:
    from jaxlib import gpu_sparse  # type: ignore

    dt = np.dtype(dtype_str)
    buf_size, opaque = gpu_sparse._cusparse.build_csr_matmat_descriptor(
        dt, dt, dt, np.dtype(np.int32), rows, cols, ncols, nnz, transpose
    )
    return int(buf_size), bytes(opaque)


def _supported_dtype(dt) -> bool:
    return jnp.dtype(dt) in (
        jnp.float32,
        jnp.float64,
        jnp.complex64,
        jnp.complex128,
    )


csr_matmat_p = core.Primitive("sparsejax_csr_matmat")


@csr_matmat_p.def_abstract_eval
def _csr_matmat_abstract_eval(data, indices, indptr, B, *, shape, transpose):
    rows, cols = shape
    out_rows = cols if transpose else rows
    return core.ShapedArray((out_rows, B.shape[1]), data.dtype)


def _csr_matmat_jax_impl(data, indices, indptr, B, *, shape, transpose):
    """Pure-JAX scatter-add fallback (used outside JIT and for unsupported dtypes)."""
    rows, cols = shape
    n_rows = rows if not transpose else cols
    # Recover row index per nnz entry from indptr (CSR -> COO row).
    nnz = data.shape[0]
    row_lengths = jnp.diff(indptr).astype(jnp.int32)
    row_idx = jnp.repeat(
        jnp.arange(rows, dtype=jnp.int32), row_lengths, total_repeat_length=nnz
    )
    if transpose:
        out_row, out_col = indices, row_idx
    else:
        out_row, out_col = row_idx, indices
    contrib = data[:, None] * B[out_col]
    out = jnp.zeros((n_rows, B.shape[1]), dtype=contrib.dtype)
    return out.at[out_row].add(contrib)


# Default impl (eager / non-JIT) routes to the same abstract-evaluable path.
mlir.register_lowering(
    csr_matmat_p, mlir.lower_fun(_csr_matmat_jax_impl, multiple_results=False)
)
dispatch.simple_impl(csr_matmat_p)


def _csr_matmat_gpu_lowering(ctx, data, indices, indptr, B, *, shape, transpose):
    rows, cols = shape
    data_aval, _, _, B_aval = ctx.avals_in
    if not _supported_dtype(data_aval.dtype):
        return mlir.lower_fun(_csr_matmat_jax_impl, multiple_results=False)(
            ctx, data, indices, indptr, B, shape=shape, transpose=transpose
        )
    nnz = data_aval.shape[0]
    Ccols = B_aval.shape[1]
    buf_size, opaque = _csr_matmat_descriptor(
        rows, cols, Ccols, nnz, bool(transpose), str(np.dtype(data_aval.dtype))
    )
    buf_aval = core.ShapedArray((buf_size,), np.int8)
    sub_ctx = ctx.replace(avals_out=[ctx.avals_out[0], buf_aval])
    rule = _jax_ffi.ffi_lowering("cusparse_csr_matmat_ffi")
    return rule(sub_ctx, data, indices, indptr, B, opaque=opaque)[:1]


def _csr_matmat_cpu_lowering(ctx, data, indices, indptr, B, *, shape, transpose):
    rows, cols = shape
    data_aval, _, _, B_aval = ctx.avals_in
    if not _supported_dtype(data_aval.dtype):
        return mlir.lower_fun(_csr_matmat_jax_impl, multiple_results=False)(
            ctx, data, indices, indptr, B, shape=shape, transpose=transpose
        )
    if transpose:
        return mlir.lower_fun(_csr_matmat_jax_impl, multiple_results=False)(
            ctx, data, indices, indptr, B, shape=shape, transpose=transpose
        )
    rule = _jax_ffi.ffi_lowering("cpu_csr_sparse_dense_ffi")
    return rule(ctx, data, indptr, indices, B)


mlir.register_lowering(csr_matmat_p, _csr_matmat_gpu_lowering, platform="cuda")
mlir.register_lowering(csr_matmat_p, _csr_matmat_cpu_lowering, platform="cpu")


def csr_matmat(
    data: jax.Array,
    indices: jax.Array,
    indptr: jax.Array,
    B: jax.Array,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
) -> jax.Array:
    return csr_matmat_p.bind(
        data, indices, indptr, B, shape=tuple(shape), transpose=bool(transpose)
    )


csr_matvec_p = core.Primitive("sparsejax_csr_matvec")


@csr_matvec_p.def_abstract_eval
def _csr_matvec_abstract_eval(data, indices, indptr, x, *, shape, transpose):
    rows, cols = shape
    out_len = cols if transpose else rows
    return core.ShapedArray((out_len,), data.dtype)


def _csr_matvec_jax_impl(data, indices, indptr, x, *, shape, transpose):
    return _csr_matmat_jax_impl(
        data, indices, indptr, x[:, None], shape=shape, transpose=transpose
    )[:, 0]


mlir.register_lowering(
    csr_matvec_p, mlir.lower_fun(_csr_matvec_jax_impl, multiple_results=False)
)
dispatch.simple_impl(csr_matvec_p)


def _csr_matvec_gpu_lowering(ctx, data, indices, indptr, x, *, shape, transpose):
    rows, cols = shape
    data_aval, _, _, x_aval = ctx.avals_in
    if not _supported_dtype(data_aval.dtype):
        return mlir.lower_fun(_csr_matvec_jax_impl, multiple_results=False)(
            ctx, data, indices, indptr, x, shape=shape, transpose=transpose
        )
    nnz = data_aval.shape[0]
    buf_size, opaque = _csr_matvec_descriptor(
        rows, cols, nnz, bool(transpose), str(np.dtype(data_aval.dtype))
    )
    buf_aval = core.ShapedArray((buf_size,), np.int8)
    sub_ctx = ctx.replace(avals_out=[ctx.avals_out[0], buf_aval])
    rule = _jax_ffi.ffi_lowering("cusparse_csr_matvec_ffi")
    return rule(sub_ctx, data, indices, indptr, x, opaque=opaque)[:1]


def _csr_matvec_cpu_lowering(ctx, data, indices, indptr, x, *, shape, transpose):
    # Reuse the matmat handler with an (n, 1) RHS — the CPU FFI accepts both.
    return (
        mlir.lower_fun(_csr_matvec_jax_impl, multiple_results=False)(
            ctx, data, indices, indptr, x, shape=shape, transpose=transpose
        )
        if transpose
        else _csr_matvec_cpu_via_matmat(ctx, data, indices, indptr, x, shape=shape)
    )


def _csr_matvec_cpu_via_matmat(ctx, data, indices, indptr, x, *, shape):
    """Lower spmv on CPU through the native ``cpu_csr_sparse_dense_ffi``."""
    return mlir.lower_fun(
        partial(_csr_matvec_cpu_native_impl, shape=shape),
        multiple_results=False,
    )(ctx, data, indices, indptr, x)


def _csr_matvec_cpu_native_impl(data, indices, indptr, x, *, shape):
    return csr_matmat(data, indices, indptr, x[:, None], shape=shape, transpose=False)[
        :, 0
    ]


mlir.register_lowering(csr_matvec_p, _csr_matvec_gpu_lowering, platform="cuda")
mlir.register_lowering(csr_matvec_p, _csr_matvec_cpu_lowering, platform="cpu")


def csr_matvec(
    data: jax.Array,
    indices: jax.Array,
    indptr: jax.Array,
    x: jax.Array,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
) -> jax.Array:
    return csr_matvec_p.bind(
        data, indices, indptr, x, shape=tuple(shape), transpose=bool(transpose)
    )
