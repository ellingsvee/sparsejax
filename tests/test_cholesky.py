from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from sparsejax import (
    SparseMatrix,
    backends,
    cholesky_solve,
    cholesky_solve_and_logdet,
    cholesky_factor,
    logdet,
)

from setup import (
    AVAILABLE_SPD_BACKENDS,
    ATOL,
    _device_for,
    _make_spd,
    _dense_to_sparse,
    _put,
)


@pytest.mark.parametrize("backend", AVAILABLE_SPD_BACKENDS)
class TestCholeskySolve:
    def test_forward(self, backend):
        device = _device_for(backend)
        A_dense = _make_spd(8)
        A = _dense_to_sparse(A_dense, device)
        b = _put(np.arange(1.0, 9.0), device)
        x = cholesky_solve(A, b, backend=backend)
        np.testing.assert_allclose(
            np.asarray(x),
            np.linalg.solve(A_dense, np.asarray(b)),
            atol=ATOL,
        )

    def test_grad(self, backend):
        device = _device_for(backend)
        A_dense = _make_spd(5)
        rows, cols = np.nonzero(np.ones_like(A_dense))
        indices = np.stack([rows, cols]).astype(np.int32)
        data = jax.device_put(jnp.asarray(A_dense[rows, cols]), device)
        b = jax.device_put(jnp.array([1.0, -0.5, 0.25, 2.0, 0.75]), device)

        def loss_sparse(d, b):
            A = SparseMatrix(data=d, indices=indices, shape=A_dense.shape)
            return jnp.sum(cholesky_solve(A, b, backend=backend) ** 2)

        def loss_dense(flat, b):
            return jnp.sum(jnp.linalg.solve(flat.reshape(A_dense.shape), b) ** 2)

        g_sp = jax.grad(loss_sparse, argnums=(0, 1))(data, b)
        g_de = jax.grad(loss_dense, argnums=(0, 1))(data, b)
        np.testing.assert_allclose(np.asarray(g_sp[0]), np.asarray(g_de[0]), atol=ATOL)
        np.testing.assert_allclose(np.asarray(g_sp[1]), np.asarray(g_de[1]), atol=ATOL)

    def test_jit(self, backend):
        device = _device_for(backend)
        A_dense = _make_spd(6, seed=3)
        A = _dense_to_sparse(A_dense, device)
        b = _put(np.ones(6), device)

        @jax.jit
        def fn(data, b):
            A2 = SparseMatrix(data=data, indices=A.indices, shape=A.shape)
            return cholesky_solve(A2, b, backend=backend)

        x = fn(A.data, b)
        np.testing.assert_allclose(
            np.asarray(x),
            np.linalg.solve(A_dense, np.asarray(b)),
            atol=ATOL,
        )


@pytest.mark.parametrize("backend", AVAILABLE_SPD_BACKENDS)
def test_cholesky_factor_reuse(backend):
    device = _device_for(backend)
    A_dense = _make_spd(7, seed=2)
    A = _dense_to_sparse(A_dense, device)
    factor = cholesky_factor(A, backend=backend)
    b1 = _put(np.arange(1.0, 8.0), device)
    b2 = _put(np.linspace(-1, 1, 7), device)
    x1 = factor(b1)
    x2 = factor(b2)
    np.testing.assert_allclose(
        np.asarray(x1), np.linalg.solve(A_dense, np.asarray(b1)), atol=ATOL
    )
    np.testing.assert_allclose(
        np.asarray(x2), np.linalg.solve(A_dense, np.asarray(b2)), atol=ATOL
    )


@pytest.mark.parametrize("backend", AVAILABLE_SPD_BACKENDS)
class TestCholeskySolveAndLogdet:
    def test_forward(self, backend):
        device = _device_for(backend)
        A_dense = _make_spd(6, seed=8)
        A = _dense_to_sparse(A_dense, device)
        b = _put(np.arange(1.0, 7.0), device)
        x, ld = cholesky_solve_and_logdet(A, b, backend=backend)
        np.testing.assert_allclose(
            np.asarray(x), np.linalg.solve(A_dense, np.asarray(b)), atol=ATOL
        )
        np.testing.assert_allclose(float(ld), np.linalg.slogdet(A_dense)[1], atol=ATOL)

    def test_grad_matches_dense(self, backend):
        device = _device_for(backend)
        A_dense = _make_spd(5, seed=9)
        rows, cols = np.nonzero(np.ones_like(A_dense))
        indices = np.stack([rows, cols]).astype(np.int32)
        data = jax.device_put(jnp.asarray(A_dense[rows, cols]), device)
        b = jax.device_put(jnp.array([1.0, -0.5, 0.25, 2.0, 0.75]), device)

        def loss_sparse(d, b):
            A = SparseMatrix(data=d, indices=indices, shape=A_dense.shape)
            x, ld = cholesky_solve_and_logdet(A, b, backend=backend)
            return jnp.sum(x**2) + 0.3 * ld

        def loss_dense(flat, b):
            mat = flat.reshape(A_dense.shape)
            x = jnp.linalg.solve(mat, b)
            _, ld = jnp.linalg.slogdet(mat)
            return jnp.sum(x**2) + 0.3 * ld

        g_sp = jax.grad(loss_sparse, argnums=(0, 1))(data, b)
        g_de = jax.grad(loss_dense, argnums=(0, 1))(data, b)
        np.testing.assert_allclose(np.asarray(g_sp[0]), np.asarray(g_de[0]), atol=ATOL)
        np.testing.assert_allclose(np.asarray(g_sp[1]), np.asarray(g_de[1]), atol=ATOL)


@pytest.mark.parametrize("backend", AVAILABLE_SPD_BACKENDS)
class TestLogdet:
    def test_forward(self, backend):
        device = _device_for(backend)
        A_dense = _make_spd(6)
        A = _dense_to_sparse(A_dense, device)
        ld = logdet(A, backend=backend)
        ref = np.linalg.slogdet(A_dense)[1]
        np.testing.assert_allclose(float(ld), ref, atol=ATOL)

    def test_grad(self, backend):
        device = _device_for(backend)
        A_dense = _make_spd(5, seed=4)
        rows, cols = np.nonzero(np.ones_like(A_dense))
        indices = np.stack([rows, cols]).astype(np.int32)
        data = jax.device_put(jnp.asarray(A_dense[rows, cols]), device)

        @jax.jit
        def fn(d):
            A = SparseMatrix(data=d, indices=indices, shape=A_dense.shape)
            return logdet(A, backend=backend)

        @jax.jit
        def fn_dense(flat):
            _, ld = jnp.linalg.slogdet(flat.reshape(A_dense.shape))
            return ld

        g_sp = jax.grad(fn)(data)
        g_de = jax.grad(fn_dense)(data)
        np.testing.assert_allclose(np.asarray(g_sp), np.asarray(g_de), atol=ATOL)


@pytest.mark.skipif(
    not backends.is_available("cholmod_takahashi"),
    reason="cholmod_takahashi backend unavailable",
)
def test_logdet_takahashi_grad_matches_cholmod_on_sparse_pattern():
    n = 8
    diag = 4.0 * np.ones(n)
    off = -1.0 * np.ones(n - 1)
    A_dense = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    rows, cols = np.nonzero(A_dense)
    indices = np.stack([rows, cols]).astype(np.int32)
    data = jnp.asarray(A_dense[rows, cols])

    @jax.jit
    def grad_takahashi(d):
        A = SparseMatrix(data=d, indices=indices, shape=A_dense.shape)
        structure = A.structure
        return jax.grad(
            lambda vals: logdet(structure.with_data(vals), backend="cholmod_takahashi")
        )(d)

    @jax.jit
    def grad_cholmod(d):
        A = SparseMatrix(data=d, indices=indices, shape=A_dense.shape)
        structure = A.structure
        return jax.grad(
            lambda vals: logdet(structure.with_data(vals), backend="cholmod")
        )(d)

    g_takahashi = grad_takahashi(data)
    g_cholmod = grad_cholmod(data)
    np.testing.assert_allclose(
        np.asarray(g_takahashi), np.asarray(g_cholmod), atol=ATOL
    )
