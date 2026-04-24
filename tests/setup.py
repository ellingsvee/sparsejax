"""Shared test helpers: backend availability, matrix construction, tolerances."""

from __future__ import annotations

from typing import cast

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from sparsejax import SparseMatrix, backends
from sparsejax.backends import BackendName


ATOL = 1e-9

ALL_SOLVE_BACKENDS = ["scipy", "cholmod", "cudss_ffi", "cudss"]
ALL_SPD_BACKENDS = ["cholmod", "cudss_ffi", "cudss"]


def _device_for(backend: str):
    if backend in ("cudss", "cudss_ffi"):
        try:
            return jax.devices("gpu")[0]
        except RuntimeError:
            return None
    return jax.devices("cpu")[0]


def _available(candidates: list[str]) -> list[str]:
    out = []
    for bk in candidates:
        if not backends.is_available(cast(BackendName, bk)):
            continue
        if bk in ("cudss", "cudss_ffi") and not any(
            d.platform == "gpu" for d in jax.devices()
        ):
            continue
        out.append(bk)
    return out


AVAILABLE_SOLVE_BACKENDS = _available(ALL_SOLVE_BACKENDS)
AVAILABLE_SPD_BACKENDS = _available(ALL_SPD_BACKENDS)


def _make_spd(n: int = 6, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    return M @ M.T + n * np.eye(n)


def _make_general(n: int = 6, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    return M + n * np.eye(n)


def _dense_to_sparse(A_dense: np.ndarray, device) -> SparseMatrix:
    rows, cols = np.nonzero(np.ones_like(A_dense))
    indices = np.stack([rows, cols]).astype(np.int32)
    data = jax.device_put(jnp.asarray(A_dense[rows, cols]), device)
    return SparseMatrix(data=data, indices=indices, shape=A_dense.shape)


def _put(x: np.ndarray, device) -> jax.Array:
    return jax.device_put(jnp.asarray(x), device)


pytestmark_skip_no_backend = pytest.mark.skipif(
    not AVAILABLE_SOLVE_BACKENDS, reason="no solve backend available"
)
