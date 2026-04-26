from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from sparsejax import SparseMatrix, eye, diag, spdiagmm, symmetrize


def test_eye_matches_dense():
    E = eye(5)
    np.testing.assert_array_equal(np.asarray(E.to_dense()), np.eye(5))
    assert E.dtype == jnp.float64
    assert E.nnz == 5


def test_eye_dtype_override():
    E = eye(3, dtype=jnp.float32)
    assert E.dtype == jnp.float32


def test_diag_matches_dense():
    d = jnp.asarray([1.0, -2.0, 3.5, 0.0])
    D = diag(d)
    np.testing.assert_allclose(np.asarray(D.to_dense()), np.diag(np.asarray(d)))


def test_diag_grad():
    d = jnp.asarray([1.0, 2.0, 3.0])

    def f(d):
        return diag(d).to_dense().sum()

    g = jax.grad(f)(d)
    np.testing.assert_allclose(np.asarray(g), np.ones(3))


def test_diag_rejects_2d():
    with pytest.raises(ValueError, match="1-D"):
        diag(jnp.zeros((2, 2)))


def test_spdiagmm_matches_dense():
    rng = np.random.default_rng(0)
    A_dense = rng.standard_normal((4, 5))
    A_dense[A_dense < 0.3] = 0.0
    A = SparseMatrix.from_dense(A_dense)
    d = jnp.asarray(rng.standard_normal(4))

    out = spdiagmm(d, A)

    expected = np.diag(np.asarray(d)) @ A_dense
    np.testing.assert_allclose(np.asarray(out.to_dense()), expected, atol=1e-12)
    assert out.structure is A.structure


def test_spdiagmm_grad():
    A = SparseMatrix.from_coo([1.0, 2.0, 3.0], [0, 1, 1], [0, 0, 1], (2, 2))
    d = jnp.asarray([0.5, -1.5])

    def f(d, data):
        Aloc = SparseMatrix(data=data, structure=A.structure)
        return spdiagmm(d, Aloc).to_dense().sum()

    gd, gv = jax.grad(f, argnums=(0, 1))(d, A.data)
    # row sums of A: row 0 -> 1.0, row 1 -> 5.0
    np.testing.assert_allclose(np.asarray(gd), [1.0, 5.0])
    # data grad: d at the row of each nnz
    np.testing.assert_allclose(np.asarray(gv), [0.5, -1.5, -1.5])


def test_spdiagmm_rejects_wrong_length():
    A = SparseMatrix.from_coo([1.0], [0], [0], (3, 3))
    with pytest.raises(ValueError, match="length 3"):
        spdiagmm(jnp.zeros(2), A)


def test_symmetrize_matches_dense():
    rng = np.random.default_rng(1)
    A_dense = rng.standard_normal((4, 4))
    A_dense[np.abs(A_dense) < 0.5] = 0.0
    A = SparseMatrix.from_dense(A_dense)

    S = symmetrize(A)

    expected = 0.5 * (A_dense + A_dense.T)
    np.testing.assert_allclose(np.asarray(S.to_dense()), expected, atol=1e-12)


def test_from_coo_sum_duplicates():
    rows = [0, 0, 1, 1, 1]
    cols = [0, 0, 1, 1, 0]
    vals = jnp.asarray([1.0, 2.0, 0.5, 0.5, 4.0])

    A = SparseMatrix.from_coo(vals, rows, cols, (2, 2), sum_duplicates=True)

    expected = np.array([[3.0, 0.0], [4.0, 1.0]])
    np.testing.assert_allclose(np.asarray(A.to_dense()), expected)
    assert A.nnz == 3


def test_from_coo_sum_duplicates_grad():
    rows = [0, 0, 1]
    cols = [0, 0, 1]

    def f(vals):
        A = SparseMatrix.from_coo(vals, rows, cols, (2, 2), sum_duplicates=True)
        return A.to_dense().sum()

    g = jax.grad(f)(jnp.asarray([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(np.asarray(g), np.ones(3))


def test_from_coo_default_unchanged():
    A = SparseMatrix.from_coo([1.0, 2.0], [0, 1], [0, 1], (2, 2))
    np.testing.assert_array_equal(np.asarray(A.to_dense()), np.diag([1.0, 2.0]))
