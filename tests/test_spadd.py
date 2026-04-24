from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from sparsejax import SparseMatrix, spadd


def test_spadd_same_pattern():
    rng = np.random.default_rng(0)
    n = 5
    dense1 = rng.standard_normal((n, n))
    dense1[dense1 < 0.3] = 0.0
    dense2 = rng.standard_normal((n, n))
    dense2[dense2 < 0.3] = 0.0
    A = SparseMatrix.from_dense(dense1)
    B = SparseMatrix.from_dense(dense2)
    C = spadd(A, B)
    np.testing.assert_allclose(np.asarray(C.to_dense()), dense1 + dense2, atol=1e-12)


def test_spadd_disjoint():
    A = SparseMatrix.from_coo([1.0], [0], [0], (2, 2))
    B = SparseMatrix.from_coo([2.0], [1], [1], (2, 2))
    C = spadd(A, B)
    expected = np.array([[1.0, 0.0], [0.0, 2.0]])
    np.testing.assert_allclose(np.asarray(C.to_dense()), expected)


def test_spadd_grad():
    rng = np.random.default_rng(3)
    n = 4
    rows, cols = np.nonzero(np.ones((n, n)))
    indices = np.stack([rows, cols]).astype(np.int32)
    d1 = jnp.asarray(rng.standard_normal(n * n))
    d2 = jnp.asarray(rng.standard_normal(n * n))

    def loss(d1, d2):
        A = SparseMatrix(data=d1, indices=indices, shape=(n, n))
        B = SparseMatrix(data=d2, indices=indices, shape=(n, n))
        C = spadd(A, B)
        return jnp.sum(C.data ** 2)

    g1, g2 = jax.grad(loss, argnums=(0, 1))(d1, d2)
    # For the union pattern with identical pairings, ∂/∂d1 = 2*(d1+d2)
    expected = 2.0 * (np.asarray(d1) + np.asarray(d2))
    np.testing.assert_allclose(np.asarray(g1), expected, atol=1e-12)
    np.testing.assert_allclose(np.asarray(g2), expected, atol=1e-12)
