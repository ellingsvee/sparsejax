from __future__ import annotations

from functools import partial
import weakref

import jax
import jax.numpy as jnp
import numpy as np

from sparsejax.sparse import SparseMatrix
from sparsejax.utils import _resolve_backend


class _PatternCacheEntry:
    __slots__ = ("indices", "a_ref", "b_ref")

    def __init__(self, indices, a_ref, b_ref):
        self.indices = indices
        self.a_ref = a_ref
        self.b_ref = b_ref


_PATTERN_CACHE: dict[tuple[int, tuple, int, tuple], _PatternCacheEntry] = {}


def _is_gpu_oom(e: BaseException) -> bool:
    name = type(e).__name__.lower()
    msg = str(e).lower()
    return "outofmemory" in name or "out of memory" in msg


def _output_index_dtype(*arrays: np.ndarray, shape: tuple) -> np.dtype:
    if max(shape) > np.iinfo(np.int32).max:
        return np.dtype(np.int64)
    if any(np.asarray(a).dtype == np.dtype(np.int64) for a in arrays):
        return np.dtype(np.int64)
    return np.dtype(np.int32)


def _drop_pattern_cache(key) -> None:
    _PATTERN_CACHE.pop(key, None)


def _dispatch_spspmm(
    backend_name: str,
    a_data,
    a_row,
    a_col,
    a_shape,
    b_data,
    b_row,
    b_col,
    b_shape,
):
    if backend_name in ("scipy", "cholmod"):
        from sparsejax.backends import scipy_backend

        return scipy_backend.spspmm(
            a_data,
            a_row,
            a_col,
            a_shape,
            b_data,
            b_row,
            b_col,
            b_shape,
        )
    if backend_name in ("cudss", "cudss_ffi"):
        try:
            from sparsejax.backends import cudss_backend

            return cudss_backend.spspmm(
                a_data,
                a_row,
                a_col,
                a_shape,
                b_data,
                b_row,
                b_col,
                b_shape,
            )
        except (ImportError, RuntimeError, MemoryError):
            from sparsejax.backends import scipy_backend

            return scipy_backend.spspmm(
                a_data,
                a_row,
                a_col,
                a_shape,
                b_data,
                b_row,
                b_col,
                b_shape,
            )
        except Exception as e:
            if not _is_gpu_oom(e):
                raise
            from sparsejax.backends import scipy_backend

            return scipy_backend.spspmm(
                a_data,
                a_row,
                a_col,
                a_shape,
                b_data,
                b_row,
                b_col,
                b_shape,
            )
    raise ValueError(f"unknown spspmm backend: {backend_name!r}")


def _resolve_spspmm_backend(A: SparseMatrix, backend: str | None) -> str:
    if backend is None or backend == "auto":
        # Dynamic-output SpGEMM currently runs through a host callback. Routing
        # that callback to CuPy can require very large cuSPARSE work buffers,
        # while SciPy only sees host arrays already materialized by the callback.
        return "scipy"
    return _resolve_backend(A, backend)


def _compute_pattern(
    a_indices: np.ndarray,
    a_shape: tuple,
    b_indices: np.ndarray,
    b_shape: tuple,
) -> np.ndarray:
    import scipy.sparse as sp

    nnz_a = a_indices.shape[1]
    nnz_b = b_indices.shape[1]
    A_pat = sp.csr_matrix(
        (
            np.ones(nnz_a, dtype=np.float64),
            (np.asarray(a_indices[0]), np.asarray(a_indices[1])),
        ),
        shape=a_shape,
    )
    B_pat = sp.csr_matrix(
        (
            np.ones(nnz_b, dtype=np.float64),
            (np.asarray(b_indices[0]), np.asarray(b_indices[1])),
        ),
        shape=b_shape,
    )
    C_pat = (A_pat @ B_pat).tocoo()
    n_cols = b_shape[1]
    lin = C_pat.row.astype(np.int64) * n_cols + C_pat.col.astype(np.int64)
    order = np.argsort(lin)
    index_dtype = _output_index_dtype(
        a_indices,
        b_indices,
        shape=(a_shape[0], b_shape[1]),
    )
    return np.stack(
        [
            np.asarray(C_pat.row, dtype=index_dtype)[order],
            np.asarray(C_pat.col, dtype=index_dtype)[order],
        ],
        axis=0,
    )


def _cached_pattern(
    a_indices: np.ndarray,
    a_shape: tuple,
    b_indices: np.ndarray,
    b_shape: tuple,
) -> np.ndarray:
    key = (id(a_indices), tuple(a_shape), id(b_indices), tuple(b_shape))
    entry = _PATTERN_CACHE.get(key)
    if entry is not None:
        a_cached = entry.a_ref() if entry.a_ref is not None else a_indices
        b_cached = entry.b_ref() if entry.b_ref is not None else b_indices
        if a_cached is a_indices and b_cached is b_indices:
            return entry.indices
        _PATTERN_CACHE.pop(key, None)

    c_indices = _compute_pattern(a_indices, a_shape, b_indices, b_shape)
    try:
        a_ref = weakref.ref(a_indices)
        weakref.finalize(a_indices, _drop_pattern_cache, key)
    except TypeError:
        a_ref = None
    try:
        b_ref = weakref.ref(b_indices)
        if b_indices is not a_indices:
            weakref.finalize(b_indices, _drop_pattern_cache, key)
    except TypeError:
        b_ref = None
    _PATTERN_CACHE[key] = _PatternCacheEntry(c_indices, a_ref, b_ref)
    return c_indices


def _pick_at_pattern(
    full_row: np.ndarray,
    full_col: np.ndarray,
    full_data: np.ndarray,
    target_row: np.ndarray,
    target_col: np.ndarray,
    n_cols: int,
    out_dtype,
) -> np.ndarray:
    nnz_target = target_row.shape[0]
    out = np.zeros(nnz_target, dtype=out_dtype)
    if nnz_target == 0 or full_data.size == 0:
        return out
    full_lin = full_row.astype(np.int64) * n_cols + full_col.astype(np.int64)
    target_lin = target_row.astype(np.int64) * n_cols + target_col.astype(np.int64)
    order = np.argsort(full_lin)
    sorted_lin = full_lin[order]
    sorted_data = np.asarray(full_data)[order]
    idx = np.searchsorted(sorted_lin, target_lin)
    valid = (idx < sorted_lin.size) & (
        sorted_lin[np.minimum(idx, sorted_lin.size - 1)] == target_lin
    )
    out[valid] = sorted_data[idx[valid]].astype(out_dtype, copy=False)
    return out


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5, 6, 7, 8))
def _spspmm_impl(
    a_data: jax.Array,
    b_data: jax.Array,
    a_indices: np.ndarray,
    a_shape: tuple,
    b_indices: np.ndarray,
    b_shape: tuple,
    c_indices: np.ndarray,
    c_shape: tuple,
    backend_name: str,
) -> jax.Array:
    out_dtype = jnp.result_type(a_data.dtype, b_data.dtype)
    nnz_c = c_indices.shape[1]
    n_cols = c_shape[1]
    a_row = np.asarray(a_indices[0])
    a_col = np.asarray(a_indices[1])
    b_row = np.asarray(b_indices[0])
    b_col = np.asarray(b_indices[1])
    c_row = np.asarray(c_indices[0])
    c_col = np.asarray(c_indices[1])

    def _host(a_h, b_h):
        cd, cr, cc, _ = _dispatch_spspmm(
            backend_name,
            np.asarray(a_h),
            a_row,
            a_col,
            a_shape,
            np.asarray(b_h),
            b_row,
            b_col,
            b_shape,
        )
        return _pick_at_pattern(
            np.asarray(cr),
            np.asarray(cc),
            np.asarray(cd),
            c_row,
            c_col,
            n_cols,
            out_dtype,
        )

    return jax.pure_callback(
        _host,
        jax.ShapeDtypeStruct((nnz_c,), out_dtype),
        a_data,
        b_data,
    )


def _spspmm_fwd(
    a_data,
    b_data,
    a_indices,
    a_shape,
    b_indices,
    b_shape,
    c_indices,
    c_shape,
    backend_name,
):
    c_data = _spspmm_impl(
        a_data,
        b_data,
        a_indices,
        a_shape,
        b_indices,
        b_shape,
        c_indices,
        c_shape,
        backend_name,
    )
    return c_data, (a_data, b_data)


def _spspmm_bwd(
    a_indices,
    a_shape,
    b_indices,
    b_shape,
    c_indices,
    c_shape,
    backend_name,
    residuals,
    dc_data,
):
    a_data, b_data = residuals
    nnz_a = a_indices.shape[1]
    nnz_b = b_indices.shape[1]
    a_dtype = a_data.dtype
    b_dtype = b_data.dtype
    a_row = np.asarray(a_indices[0])
    a_col = np.asarray(a_indices[1])
    b_row = np.asarray(b_indices[0])
    b_col = np.asarray(b_indices[1])
    c_row = np.asarray(c_indices[0])
    c_col = np.asarray(c_indices[1])
    K = a_shape[1]  # inner dim; columns of A == rows of B

    def _host(a_h, b_h, dc_h):
        a_np = np.asarray(a_h)
        b_np = np.asarray(b_h)
        dc_np = np.asarray(dc_h)

        da_full_data, da_full_row, da_full_col, _ = _dispatch_spspmm(
            backend_name,
            dc_np,
            c_row,
            c_col,
            c_shape,
            b_np,
            b_col,
            b_row,
            (b_shape[1], b_shape[0]),
        )
        da_data = _pick_at_pattern(
            np.asarray(da_full_row),
            np.asarray(da_full_col),
            np.asarray(da_full_data),
            a_row,
            a_col,
            K,
            a_dtype,
        )

        db_full_data, db_full_row, db_full_col, _ = _dispatch_spspmm(
            backend_name,
            a_np,
            a_col,
            a_row,
            (a_shape[1], a_shape[0]),
            dc_np,
            c_row,
            c_col,
            c_shape,
        )
        db_data = _pick_at_pattern(
            np.asarray(db_full_row),
            np.asarray(db_full_col),
            np.asarray(db_full_data),
            b_row,
            b_col,
            b_shape[1],
            b_dtype,
        )

        return da_data, db_data

    return jax.pure_callback(
        _host,
        (
            jax.ShapeDtypeStruct((nnz_a,), a_dtype),
            jax.ShapeDtypeStruct((nnz_b,), b_dtype),
        ),
        a_data,
        b_data,
        dc_data,
    )


_spspmm_impl.defvjp(_spspmm_fwd, _spspmm_bwd)


def spspmm(
    A: SparseMatrix,
    B: SparseMatrix,
    *,
    backend: str | None = "auto",
) -> SparseMatrix:
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"shape mismatch: A {A.shape} @ B {B.shape}")
    backend_name = _resolve_spspmm_backend(A, backend)
    c_shape = (A.shape[0], B.shape[1])
    c_indices = _cached_pattern(A.indices, A.shape, B.indices, B.shape)
    c_data = _spspmm_impl(
        A.data,
        B.data,
        A.indices,
        A.shape,
        B.indices,
        B.shape,
        c_indices,
        c_shape,
        backend_name,
    )
    return SparseMatrix(data=c_data, indices=c_indices, shape=c_shape)
