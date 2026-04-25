from __future__ import annotations

import weakref

import jax
import jax.numpy as jnp
import numpy as np


def _require_cholmod():
    try:
        from sksparse.cholmod import cho_factor as _cho_factor  # type: ignore
    except ImportError as e:  # pragma: no cover - env-dependent
        raise RuntimeError(
            "cholmod backend requires scikit-sparse. "
            "Install with `uv pip install scikit-sparse`."
        ) from e
    return _cho_factor


class _CachedFactor:
    __slots__ = ("factor", "shape", "indices_ref")

    def __init__(self, factor, shape, indices_ref):
        self.factor = factor
        self.shape = shape
        # Weak ref to the indices ndarray; strong ref would pin it for
        # the process lifetime and defeat weakref-finalize cleanup.
        self.indices_ref = indices_ref


# Map id(indices) -> _CachedFactor. Strong-ref dict (a previous
# WeakValueDictionary design dropped entries instantly because nothing
# else held a strong reference, silently invalidating the cache on every
# call). Lifetime is tied to the indices array via weakref.finalize.
_FACTOR_CACHE: dict[int, _CachedFactor] = {}


def _on_indices_collected(key: int) -> None:
    _FACTOR_CACHE.pop(key, None)


def _get_or_build_factor(data_np: np.ndarray, indices: np.ndarray, shape):
    """Return a cholmod factor for (indices, shape), numerically refreshed to
    ``data_np``. Symbolic analysis is cached by ``id(indices)``."""
    import scipy.sparse as sp

    cho_factor = _require_cholmod()

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
    # scikit-sparse may need writable buffers internally — callback inputs
    # are read-only, so copy once here.
    data_w = np.array(data_np, copy=True)
    A = sp.coo_matrix((data_w, (indices[0], indices[1])), shape=shape).tocsc()
    if entry is None or entry.shape != shape:
        factor = cho_factor(A)
        try:
            indices_ref = weakref.ref(indices)
        except TypeError:
            indices_ref = None
        entry = _CachedFactor(factor, shape, indices_ref)
        _FACTOR_CACHE[key] = entry
        try:
            weakref.finalize(indices, _on_indices_collected, key)
        except TypeError:
            # Object isn't weakref-able — fall back to leaking the entry.
            pass
    else:
        # numerical refactorization on the cached symbolic ordering
        entry.factor.factorize(A)
    return entry.factor


def cholesky_solve(
    data: jax.Array,
    indices: np.ndarray,
    shape: tuple[int, int],
    b: jax.Array,
) -> jax.Array:
    out_shape = b.shape
    out_dtype = jnp.result_type(data.dtype, b.dtype)

    def _host(data_h, b_h):
        factor = _get_or_build_factor(np.asarray(data_h), indices, shape)
        x = factor.solve(np.array(b_h, copy=True))
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


def build_factor(data: np.ndarray, indices: np.ndarray, shape):
    from sparsejax.ops.cholesky import CholeskyFactor

    factor = _get_or_build_factor(np.asarray(data), indices, shape)
    ld = float(factor.logdet())

    def _solve(b):
        return jnp.asarray(factor.solve(np.array(b, copy=True)))

    return CholeskyFactor(solve_fn=_solve, logdet_val=ld)
