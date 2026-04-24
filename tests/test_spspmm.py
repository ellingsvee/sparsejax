from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from sparsejax import SparseMatrix, spspmm


def _backends():
    out = ["scipy"]
    try:
        if any(d.platform == "gpu" for d in jax.devices()):
            try:
                import cupy  # noqa: F401

                out.append("cudss")
            except ImportError:
                pass
    except RuntimeError:
        pass
    return out


@pytest.mark.parametrize("backend", _backends())
def test_spspmm_square(backend):
    rng = np.random.default_rng(0)
    n = 6
    A = rng.standard_normal((n, n))
    A[A < 0.4] = 0
    B = rng.standard_normal((n, n))
    B[B < 0.4] = 0
    As = SparseMatrix.from_dense(A)
    Bs = SparseMatrix.from_dense(B)
    C = spspmm(As, Bs, backend=backend)
    np.testing.assert_allclose(np.asarray(C.to_dense()), A @ B, atol=1e-10)


@pytest.mark.parametrize("backend", _backends())
def test_spspmm_rectangular(backend):
    rng = np.random.default_rng(1)
    A = rng.standard_normal((4, 6))
    A[A < 0.3] = 0
    B = rng.standard_normal((6, 3))
    B[B < 0.3] = 0
    As = SparseMatrix.from_dense(A)
    Bs = SparseMatrix.from_dense(B)
    C = spspmm(As, Bs, backend=backend)
    assert C.shape == (4, 3)
    np.testing.assert_allclose(np.asarray(C.to_dense()), A @ B, atol=1e-10)
