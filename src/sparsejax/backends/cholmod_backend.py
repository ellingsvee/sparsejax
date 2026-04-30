from __future__ import annotations

import hashlib
import weakref
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class _CholmodApi:
    name: str
    factorize: Callable


_CHOLMOD_API: _CholmodApi | None = None


def _require_cholmod():
    global _CHOLMOD_API
    if _CHOLMOD_API is not None:
        return _CHOLMOD_API

    try:
        from sksparse.cholmod import cho_factor as _cho_factor  # type: ignore
    except ImportError:  # pragma: no cover - env-dependent
        try:
            from sksparse.cholmod import cholesky as _cholesky  # type: ignore
        except ImportError as legacy_api_error:
            raise RuntimeError(
                "cholmod backend requires scikit-sparse. "
                "Install with `uv pip install scikit-sparse`."
            ) from legacy_api_error
        _CHOLMOD_API = _CholmodApi("legacy", _cholesky)
    else:
        _CHOLMOD_API = _CholmodApi("new", _cho_factor)
    return _CHOLMOD_API


def _refactorize(api: _CholmodApi, factor, A):
    if api.name == "new":
        factor.factorize(A)
    else:
        factor.cholesky_inplace(A)


def _solve_factor(api: _CholmodApi, factor, b):
    if api.name == "new":
        return factor.solve(b)
    return factor.solve_A(b)


def _factor_l(api: _CholmodApi, factor):
    if api.name == "new":
        return factor.L
    return factor.L()


def _factor_perm(api: _CholmodApi, factor) -> np.ndarray:
    if api.name == "new":
        return np.asarray(factor.get_perm(), dtype=np.int32)
    return np.asarray(factor.P(), dtype=np.int32)


def _coo_from_upper(api: _CholmodApi, data, row, col, upper, shape):
    import scipy.sparse as sp

    if api.name == "new":
        # scikit-sparse >= 0.5.0 uses the upper triangle by default.
        return sp.coo_matrix(
            (data, (row[upper], col[upper])),
            shape=shape,
        ).tocsc()

    # Older scikit-sparse uses the lower triangle of the matrix it receives.
    # Callers expose the active SPD structure as an upper-triangle view, so
    # transpose that view into the lower triangle for the legacy API.
    return sp.coo_matrix(
        (data, (col[upper], row[upper])),
        shape=shape,
    ).tocsc()


def _factorize_with_fallback(api: _CholmodApi, make_A):
    global _CHOLMOD_API

    A = make_A(api)
    try:
        return api.factorize(A), api
    except AttributeError as e:
        if api.name != "new":
            raise
        try:
            from sksparse.cholmod import cholesky as _cholesky  # type: ignore
        except ImportError:
            raise e
        fallback_api = _CholmodApi("legacy", _cholesky)
        _CHOLMOD_API = fallback_api
        return fallback_api.factorize(make_A(fallback_api)), fallback_api
    except ImportError as e:  # pragma: no cover - env-dependent
        raise RuntimeError(
            "cholmod backend requires scikit-sparse. "
            "Install with `uv pip install scikit-sparse`."
        ) from e


class _CachedFactor:
    __slots__ = ("api", "factor", "shape", "indices_ref", "data_digest")

    def __init__(self, api, factor, shape, indices_ref, data_digest):
        self.api = api
        self.factor = factor
        self.shape = shape
        # Weak ref to the indices ndarray; strong ref would pin it for
        # the process lifetime and defeat weakref-finalize cleanup.
        self.indices_ref = indices_ref
        self.data_digest = data_digest


# Map id(indices) -> _CachedFactor. Strong-ref dict (a previous
# WeakValueDictionary design dropped entries instantly because nothing
# else held a strong reference, silently invalidating the cache on every
# call). Lifetime is tied to the indices array via weakref.finalize.
_FACTOR_CACHE: dict[int, _CachedFactor] = {}


def _on_indices_collected(key: int) -> None:
    _FACTOR_CACHE.pop(key, None)


def _numeric_digest(data: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(data)
    h = hashlib.blake2b(digest_size=16)
    h.update(arr.dtype.str.encode())
    h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    h.update(arr.view(np.uint8))
    return h.digest()


def _get_or_build_factor_entry(data_np: np.ndarray, indices: np.ndarray, shape):
    """Return a cholmod factor for (indices, shape), numerically refreshed to
    ``data_np``. Symbolic analysis is cached by ``id(indices)``."""
    api = _require_cholmod()

    key = id(indices)
    entry = _FACTOR_CACHE.get(key)
    # Reject stale entries from id reuse: if the cached weakref no longer
    # resolves to `indices`, treat as a miss. `indices_ref is None` means
    # the original wasn't weakref-able; we can't verify identity in that
    # case and accept the small id-reuse risk.
    if entry is not None and entry.indices_ref is not None:
        cached = entry.indices_ref()
        if cached is not indices:
            _FACTOR_CACHE.pop(key, None)
            entry = None
    # CHOLMOD's default factorization uses only the upper triangle. Build that
    # view directly so full symmetric inputs do not pay to materialize a second
    # triangle on the host.
    row = np.asarray(indices[0])
    col = np.asarray(indices[1])
    upper = row <= col
    # scikit-sparse may need writable buffers internally — callback inputs are
    # read-only, so copy once here.
    # CHOLMOD is materially more robust in double precision. Keep this cast at
    # the backend boundary so callers can still run the surrounding JAX graph in
    # float32 and receive float32 outputs.
    data_w = np.array(np.asarray(data_np)[upper], dtype=np.float64, copy=True)
    data_digest = _numeric_digest(data_w)
    if (
        entry is not None
        and entry.api == api
        and entry.shape == shape
        and entry.data_digest == data_digest
    ):
        return entry

    if entry is not None and entry.api != api:
        entry = None

    def make_A(api: _CholmodApi):
        return _coo_from_upper(api, data_w, row, col, upper, shape)

    if entry is None or entry.shape != shape:
        factor, api = _factorize_with_fallback(api, make_A)
        try:
            indices_ref = weakref.ref(indices)
        except TypeError:
            indices_ref = None
        entry = _CachedFactor(api, factor, shape, indices_ref, data_digest)
        _FACTOR_CACHE[key] = entry
        try:
            weakref.finalize(indices, _on_indices_collected, key)
        except TypeError:
            # Object isn't weakref-able — fall back to leaking the entry.
            pass
    else:
        # numerical refactorization on the cached symbolic ordering
        _refactorize(entry.api, entry.factor, make_A(entry.api))
        entry.data_digest = data_digest
    return entry


def _get_or_build_factor(data_np: np.ndarray, indices: np.ndarray, shape):
    return _get_or_build_factor_entry(data_np, indices, shape).factor


def cholesky_solve(
    data: jax.Array,
    indices: np.ndarray,
    shape: tuple[int, int],
    b: jax.Array,
) -> jax.Array:
    out_shape = b.shape
    out_dtype = jnp.result_type(data.dtype, b.dtype)

    def _host(data_h, b_h):
        entry = _get_or_build_factor_entry(np.asarray(data_h), indices, shape)
        x = _solve_factor(
            entry.api,
            entry.factor,
            np.array(b_h, dtype=np.float64, copy=True),
        )
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


def cholesky_solve_and_logdet(
    data: jax.Array,
    indices: np.ndarray,
    shape: tuple[int, int],
    b: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    x_shape = b.shape
    out_dtype = jnp.result_type(data.dtype, b.dtype)
    ld_dtype = data.dtype

    def _host(data_h, b_h):
        entry = _get_or_build_factor_entry(np.asarray(data_h), indices, shape)
        x = _solve_factor(
            entry.api,
            entry.factor,
            np.array(b_h, dtype=np.float64, copy=True),
        )
        x = np.asarray(x).astype(out_dtype, copy=False)
        if x.shape != x_shape:
            x = x.reshape(x_shape)
        ld = np.asarray(entry.factor.logdet(), dtype=ld_dtype)
        return x, ld

    return jax.pure_callback(
        _host,
        (
            jax.ShapeDtypeStruct(x_shape, out_dtype),
            jax.ShapeDtypeStruct((), ld_dtype),
        ),
        data,
        b,
    )


def logdet(
    data: jax.Array,
    indices: np.ndarray,
    shape: tuple[int, int],
) -> jax.Array:
    out_dtype = data.dtype

    def _host(data_h):
        factor = _get_or_build_factor(np.asarray(data_h), indices, shape)
        val = factor.logdet()
        return np.asarray(val, dtype=out_dtype)

    return jax.pure_callback(_host, jax.ShapeDtypeStruct((), out_dtype), data)


def selected_inverse_entries_takahashi(
    data: jax.Array,
    indices: np.ndarray,
    shape: tuple[int, int],
) -> jax.Array:
    """Selected inverse values aligned with ``indices`` via CHOLMOD + Rust.

    Returns ``A^{-1}[row[k], col[k]]`` for each input COO entry. This is used
    as an opt-in logdet VJP path to avoid solving against all unique sparse
    columns.
    """
    out_dtype = data.dtype
    out_shape = (indices.shape[1],)
    row = np.asarray(indices[0], dtype=np.int64)
    col = np.asarray(indices[1], dtype=np.int64)

    def _host(data_h):
        try:
            from sparsejax.rust_backend import takahashi_masked
        except ImportError as e:  # pragma: no cover - optional extension
            raise RuntimeError(
                "backend='cholmod_takahashi' requires the optional Rust extension. "
                "Build it with `uv run --extra rust maturin develop --release`."
            ) from e

        entry = _get_or_build_factor_entry(np.asarray(data_h), indices, shape)
        L = _factor_l(entry.api, entry.factor)
        perm = _factor_perm(entry.api, entry.factor)
        gathered = takahashi_masked(
            np.asarray(L.indptr, dtype=np.int32),
            np.asarray(L.indices, dtype=np.int32),
            np.asarray(L.data, dtype=np.float64),
            perm,
            row.astype(np.int32, copy=False),
            col.astype(np.int32, copy=False),
            int(shape[0]),
        )
        return np.asarray(gathered).astype(out_dtype, copy=False)

    return jax.pure_callback(
        _host,
        jax.ShapeDtypeStruct(out_shape, out_dtype),
        data,
    )


def build_factor(data: np.ndarray, indices: np.ndarray, shape):
    from sparsejax.ops.cholesky import CholeskyFactor

    entry = _get_or_build_factor_entry(np.asarray(data), indices, shape)
    ld = float(entry.factor.logdet())

    def solve(b):
        return jnp.asarray(
            _solve_factor(
                entry.api,
                entry.factor,
                np.array(b, dtype=np.float64, copy=True),
            )
        )

    return CholeskyFactor(solve_fn=solve, logdet_val=ld)
