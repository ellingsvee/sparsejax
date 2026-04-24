from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from sparsejax import SparseMatrix, spdmm

from setup import ATOL, _make_general, _dense_to_sparse, _put


def _devices():
    devs = [jax.devices("cpu")[0]]
    try:
        devs.append(jax.devices("gpu")[0])
    except RuntimeError:
        pass
    return devs


@pytest.mark.parametrize("device", _devices())
class TestSpdmm:
    def test_forward_1d(self, device):
        A_dense = _make_general(8)
        A = _dense_to_sparse(A_dense, device)
        x = _put(np.arange(1.0, 9.0), device)
        y = spdmm(A, x)
        np.testing.assert_allclose(np.asarray(y), A_dense @ np.asarray(x), atol=ATOL)

    def test_forward_2d(self, device):
        A_dense = _make_general(7, seed=1)
        A = _dense_to_sparse(A_dense, device)
        rng = np.random.default_rng(0)
        X = _put(rng.standard_normal((7, 5)), device)
        Y = spdmm(A, X)
        np.testing.assert_allclose(np.asarray(Y), A_dense @ np.asarray(X), atol=ATOL)

    def test_transpose(self, device):
        A_dense = _make_general(6, seed=2)
        A = _dense_to_sparse(A_dense, device)
        rng = np.random.default_rng(1)
        X = _put(rng.standard_normal((6, 3)), device)
        Y = spdmm(A, X, transpose=True)
        np.testing.assert_allclose(
            np.asarray(Y), A_dense.T @ np.asarray(X), atol=ATOL
        )

    def test_grad(self, device):
        A_dense = _make_general(5, seed=4)
        rows, cols = np.nonzero(np.ones_like(A_dense))
        indices = np.stack([rows, cols]).astype(np.int32)
        data = jax.device_put(jnp.asarray(A_dense[rows, cols]), device)
        rng = np.random.default_rng(7)
        X = jax.device_put(jnp.asarray(rng.standard_normal((5, 4))), device)

        def loss_sparse(data, X):
            A = SparseMatrix(data=data, indices=indices, shape=A_dense.shape)
            return jnp.sum(spdmm(A, X) ** 2)

        def loss_dense(flat, X):
            return jnp.sum((flat.reshape(A_dense.shape) @ X) ** 2)

        g_sp = jax.grad(loss_sparse, argnums=(0, 1))(data, X)
        g_de = jax.grad(loss_dense, argnums=(0, 1))(data, X)
        np.testing.assert_allclose(np.asarray(g_sp[0]), np.asarray(g_de[0]), atol=ATOL)
        np.testing.assert_allclose(np.asarray(g_sp[1]), np.asarray(g_de[1]), atol=ATOL)

    def test_jit(self, device):
        A_dense = _make_general(6, seed=5)
        A = _dense_to_sparse(A_dense, device)
        rng = np.random.default_rng(3)
        X = _put(rng.standard_normal((6, 4)), device)

        @jax.jit
        def fn(data, X):
            A2 = SparseMatrix(data=data, indices=A.indices, shape=A.shape)
            return spdmm(A2, X)

        Y = fn(A.data, X)
        np.testing.assert_allclose(np.asarray(Y), A_dense @ np.asarray(X), atol=ATOL)
