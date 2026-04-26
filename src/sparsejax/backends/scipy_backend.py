from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np


def _to_csr(data_np, row, col, shape):
    import scipy.sparse as sp

    return sp.coo_matrix((data_np, (row, col)), shape=shape).tocsr()


def solve(
    data, row, col, shape: Tuple[int, int], b, *, method: str = "lu"
) -> jax.Array:
    out_shape = b.shape
    out_dtype = jnp.result_type(data.dtype, b.dtype)

    def _host(data_h, b_h):
        import scipy.sparse.linalg as spla

        A = _to_csr(np.asarray(data_h), row, col, shape)
        b_np = np.asarray(b_h)
        if method == "lu":
            x = spla.spsolve(A, b_np)
        elif method == "cg":
            if b_np.ndim == 2:
                cols = [spla.cg(A, b_np[:, k])[0] for k in range(b_np.shape[1])]
                x = np.column_stack(cols)
            else:
                x, _ = spla.cg(A, b_np)
        else:
            raise ValueError(f"unknown scipy solve method: {method!r}")
        x = np.asarray(x).astype(out_dtype, copy=False)
        if x.shape != out_shape:
            x = x.reshape(out_shape)
        return x

    return jax.pure_callback(
        _host,
        jax.ShapeDtypeStruct(out_shape, out_dtype),
        data,
        b,
    )


def solve_and_logdet(
    data, indices, shape: Tuple[int, int], b
) -> tuple[jax.Array, jax.Array]:
    out_shape = b.shape
    out_dtype = jnp.result_type(data.dtype, b.dtype)
    ld_dtype = data.dtype
    row = np.asarray(indices[0])
    col = np.asarray(indices[1])

    def _host(data_h, b_h):
        import scipy.sparse.linalg as spla

        A = _to_csr(np.asarray(data_h), row, col, shape)
        b_np = np.asarray(b_h)
        lu = spla.splu(A.tocsc())
        x = lu.solve(b_np)
        diag_u = lu.U.diagonal()
        sign = float(lu.perm_r.size and 1.0)
        if np.any(diag_u == 0):
            ld = np.nan
        else:
            ld = np.sum(np.log(np.abs(diag_u)))
            if sign <= 0:
                ld = np.nan
        x = np.asarray(x).astype(out_dtype, copy=False)
        if x.shape != out_shape:
            x = x.reshape(out_shape)
        return x, np.asarray(ld, dtype=ld_dtype)

    return jax.pure_callback(
        _host,
        (
            jax.ShapeDtypeStruct(out_shape, out_dtype),
            jax.ShapeDtypeStruct((), ld_dtype),
        ),
        data,
        b,
    )


def logdet(data, indices, shape: Tuple[int, int]) -> jax.Array:
    """Log-determinant via dense LU (only suitable for small matrices)."""
    row = np.asarray(indices[0])
    col = np.asarray(indices[1])
    out_dtype = data.dtype

    def _host(data_h):
        A = _to_csr(np.asarray(data_h), row, col, shape).toarray()
        sign, ld = np.linalg.slogdet(A)
        if sign <= 0:
            # not SPD / singular — return NaN, let user notice
            return np.asarray(np.nan, dtype=out_dtype)
        return np.asarray(ld, dtype=out_dtype)

    return jax.pure_callback(_host, jax.ShapeDtypeStruct((), out_dtype), data)


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
    import scipy.sparse as sp

    A = sp.coo_matrix((np.asarray(a_data), (a_row, a_col)), shape=a_shape).tocsr()
    B = sp.coo_matrix((np.asarray(b_data), (b_row, b_col)), shape=b_shape).tocsr()
    C = (A @ B).tocoo()
    return (
        np.asarray(C.data),
        np.asarray(C.row, dtype=np.int32),
        np.asarray(C.col, dtype=np.int32),
        (a_shape[0], b_shape[1]),
    )
