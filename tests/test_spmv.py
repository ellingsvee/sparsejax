from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from sparsejax import SparseMatrix, spmv

from setup import ATOL, _make_general, _dense_to_sparse, _put


def _devices():
    devs = [jax.devices("cpu")[0]]
    try:
        devs.append(jax.devices("gpu")[0])
    except RuntimeError:
        pass
    return devs


@pytest.mark.parametrize("device", _devices())
class TestSpmv:
    def test_forward(self, device):
        A_dense = _make_general(9)
        A = _dense_to_sparse(A_dense, device)
        x = _put(np.linspace(-1, 1, 9), device)
        y = spmv(A, x)
        np.testing.assert_allclose(np.asarray(y), A_dense @ np.asarray(x), atol=ATOL)

    def test_transpose(self, device):
        A_dense = _make_general(7, seed=3)
        A = _dense_to_sparse(A_dense, device)
        x = _put(np.arange(1.0, 8.0), device)
        y = spmv(A, x, transpose=True)
        np.testing.assert_allclose(np.asarray(y), A_dense.T @ np.asarray(x), atol=ATOL)

    def test_multi_rhs(self, device):
        A_dense = _make_general(6, seed=5)
        A = _dense_to_sparse(A_dense, device)
        rng = np.random.default_rng(2)
        X = _put(rng.standard_normal((6, 4)), device)
        Y = spmv(A, X)
        np.testing.assert_allclose(np.asarray(Y), A_dense @ np.asarray(X), atol=ATOL)

    def test_grad(self, device):
        A_dense = _make_general(5, seed=7)
        rows, cols = np.nonzero(np.ones_like(A_dense))
        indices = np.stack([rows, cols]).astype(np.int32)
        data = jax.device_put(jnp.asarray(A_dense[rows, cols]), device)
        x = jax.device_put(jnp.arange(1.0, 6.0), device)

        def loss_sparse(data, x):
            A = SparseMatrix(data=data, indices=indices, shape=A_dense.shape)
            return jnp.sum(spmv(A, x) ** 2)

        def loss_dense(flat, x):
            return jnp.sum((flat.reshape(A_dense.shape) @ x) ** 2)

        g_sp = jax.grad(loss_sparse, argnums=(0, 1))(data, x)
        g_de = jax.grad(loss_dense, argnums=(0, 1))(data, x)
        np.testing.assert_allclose(np.asarray(g_sp[0]), np.asarray(g_de[0]), atol=ATOL)
        np.testing.assert_allclose(np.asarray(g_sp[1]), np.asarray(g_de[1]), atol=ATOL)

    def test_jit(self, device):
        A_dense = _make_general(5)
        A = _dense_to_sparse(A_dense, device)
        x = _put(np.ones(5), device)

        @jax.jit
        def fn(data, x):
            A2 = SparseMatrix(data=data, indices=A.indices, shape=A.shape)
            return spmv(A2, x)

        y = fn(A.data, x)
        np.testing.assert_allclose(np.asarray(y), A_dense @ np.asarray(x), atol=ATOL)
