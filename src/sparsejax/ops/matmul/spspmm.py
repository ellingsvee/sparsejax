from __future__ import annotations

from functools import partial
import weakref

import jax
import jax.numpy as jnp
import numpy as np

from sparsejax._csr import coo_to_csr
from sparsejax.dense_mode import is_dense_mode
from sparsejax.sparse import SparseMatrix
from sparsejax.utils import _resolve_backend


class _PatternCacheEntry:
    __slots__ = ("indices", "a_ref", "b_ref")

    def __init__(self, indices, a_ref, b_ref):
        self.indices = indices
        self.a_ref = a_ref
        self.b_ref = b_ref


class _StaticPlanCacheEntry:
    __slots__ = ("c_pos", "a_pos", "b_pos", "a_ref", "b_ref", "c_ref")

    def __init__(self, c_pos, a_pos, b_pos, a_ref, b_ref, c_ref):
        self.c_pos = c_pos
        self.a_pos = a_pos
        self.b_pos = b_pos
        self.a_ref = a_ref
        self.b_ref = b_ref
        self.c_ref = c_ref


_PATTERN_CACHE: dict[tuple[int, tuple, int, tuple], _PatternCacheEntry] = {}
_STATIC_PLAN_CACHE: dict[tuple[int, tuple, int, tuple, int], _StaticPlanCacheEntry] = {}


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


def _drop_static_plan_cache(key) -> None:
    _STATIC_PLAN_CACHE.pop(key, None)


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


def _spspmm_scipy_csr(
    a_data,
    a_indptr: np.ndarray,
    a_indices: np.ndarray,
    a_shape: tuple,
    b_data,
    b_indptr: np.ndarray,
    b_indices: np.ndarray,
    b_shape: tuple,
):
    import scipy.sparse as sp

    A = sp.csr_matrix((np.asarray(a_data), a_indices, a_indptr), shape=a_shape)
    B = sp.csr_matrix((np.asarray(b_data), b_indices, b_indptr), shape=b_shape)
    if not A.has_canonical_format:
        A.sum_duplicates()
    if not B.has_canonical_format:
        B.sum_duplicates()
    C = (A @ B).tocoo()
    index_dtype = _output_index_dtype(
        a_indices, b_indices, shape=(a_shape[0], b_shape[1])
    )
    return (
        np.asarray(C.data),
        np.asarray(C.row, dtype=index_dtype),
        np.asarray(C.col, dtype=index_dtype),
        (a_shape[0], b_shape[1]),
    )


def _resolve_spspmm_backend(A: SparseMatrix, backend: str | None) -> str:
    if backend is None or backend == "auto":
        return "static"
    if backend == "static":
        return backend
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


def _compute_static_plan(
    a_indices: np.ndarray,
    a_shape: tuple,
    b_indices: np.ndarray,
    b_shape: tuple,
    c_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    del a_shape
    nnz_b = b_indices.shape[1]
    inner_dim = b_shape[0]
    n_cols = b_shape[1]
    b_by_row: list[list[int]] = [[] for _ in range(inner_dim)]
    b_row = np.asarray(b_indices[0], dtype=np.int64)
    b_col = np.asarray(b_indices[1], dtype=np.int64)
    for b_pos in range(nnz_b):
        b_by_row[int(b_row[b_pos])].append(b_pos)

    c_row = np.asarray(c_indices[0], dtype=np.int64)
    c_col = np.asarray(c_indices[1], dtype=np.int64)
    c_lookup = {
        int(row) * n_cols + int(col): pos
        for pos, (row, col) in enumerate(zip(c_row, c_col))
    }

    c_pos_out: list[int] = []
    a_pos_out: list[int] = []
    b_pos_out: list[int] = []
    a_row = np.asarray(a_indices[0], dtype=np.int64)
    a_col = np.asarray(a_indices[1], dtype=np.int64)
    for a_pos, (row, inner) in enumerate(zip(a_row, a_col)):
        for b_pos in b_by_row[int(inner)]:
            out_pos = c_lookup[int(row) * n_cols + int(b_col[b_pos])]
            c_pos_out.append(out_pos)
            a_pos_out.append(a_pos)
            b_pos_out.append(b_pos)

    index_dtype = (
        np.int64
        if max(len(c_pos_out), a_indices.shape[1], nnz_b) > np.iinfo(np.int32).max
        else np.int32
    )
    return (
        np.asarray(c_pos_out, dtype=index_dtype),
        np.asarray(a_pos_out, dtype=index_dtype),
        np.asarray(b_pos_out, dtype=index_dtype),
    )


def _cached_static_plan(
    a_indices: np.ndarray,
    a_shape: tuple,
    b_indices: np.ndarray,
    b_shape: tuple,
    c_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = (id(a_indices), tuple(a_shape), id(b_indices), tuple(b_shape), id(c_indices))
    entry = _STATIC_PLAN_CACHE.get(key)
    if entry is not None:
        a_cached = entry.a_ref() if entry.a_ref is not None else a_indices
        b_cached = entry.b_ref() if entry.b_ref is not None else b_indices
        c_cached = entry.c_ref() if entry.c_ref is not None else c_indices
        if a_cached is a_indices and b_cached is b_indices and c_cached is c_indices:
            return entry.c_pos, entry.a_pos, entry.b_pos
        _STATIC_PLAN_CACHE.pop(key, None)

    c_pos, a_pos, b_pos = _compute_static_plan(
        a_indices, a_shape, b_indices, b_shape, c_indices
    )
    try:
        a_ref = weakref.ref(a_indices)
        weakref.finalize(a_indices, _drop_static_plan_cache, key)
    except TypeError:
        a_ref = None
    try:
        b_ref = weakref.ref(b_indices)
        if b_indices is not a_indices:
            weakref.finalize(b_indices, _drop_static_plan_cache, key)
    except TypeError:
        b_ref = None
    try:
        c_ref = weakref.ref(c_indices)
        weakref.finalize(c_indices, _drop_static_plan_cache, key)
    except TypeError:
        c_ref = None
    _STATIC_PLAN_CACHE[key] = _StaticPlanCacheEntry(
        c_pos, a_pos, b_pos, a_ref, b_ref, c_ref
    )
    return c_pos, a_pos, b_pos


def _spspmm_static_impl(
    a_data: jax.Array,
    b_data: jax.Array,
    nnz_c: int,
    c_pos: np.ndarray,
    a_pos: np.ndarray,
    b_pos: np.ndarray,
) -> jax.Array:
    out_dtype = jnp.result_type(a_data.dtype, b_data.dtype)
    if c_pos.size == 0:
        return jnp.zeros((nnz_c,), dtype=out_dtype)
    c_pos_j = jnp.asarray(c_pos)
    a_pos_j = jnp.asarray(a_pos)
    b_pos_j = jnp.asarray(b_pos)
    products = a_data[a_pos_j] * b_data[b_pos_j]
    return jnp.zeros((nnz_c,), dtype=out_dtype).at[c_pos_j].add(products)


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
    use_scipy_csr = backend_name in ("scipy", "cholmod")
    if use_scipy_csr:
        a_csr = coo_to_csr(a_indices, a_shape)
        b_csr = coo_to_csr(b_indices, b_shape)
        a_callback_data = a_data if a_csr.order_is_identity else a_data[a_csr.order]
        b_callback_data = b_data if b_csr.order_is_identity else b_data[b_csr.order]
    else:
        a_callback_data = a_data
        b_callback_data = b_data

    def _host(a_h, b_h):
        if use_scipy_csr:
            cd, cr, cc, _ = _spspmm_scipy_csr(
                a_h,
                a_csr.indptr,
                a_csr.col_idx,
                a_shape,
                b_h,
                b_csr.indptr,
                b_csr.col_idx,
                b_shape,
            )
        else:
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
        a_callback_data,
        b_callback_data,
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
    if is_dense_mode():
        Cd = A.to_dense() @ B.to_dense()
        c_row = jnp.asarray(c_indices[0])
        c_col = jnp.asarray(c_indices[1])
        return SparseMatrix(data=Cd[c_row, c_col], indices=c_indices, shape=c_shape)
    if backend_name == "static":
        c_pos, a_pos, b_pos = _cached_static_plan(
            A.indices, A.shape, B.indices, B.shape, c_indices
        )
        c_data = _spspmm_static_impl(
            A.data, B.data, c_indices.shape[1], c_pos, a_pos, b_pos
        )
        return SparseMatrix(data=c_data, indices=c_indices, shape=c_shape)
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
