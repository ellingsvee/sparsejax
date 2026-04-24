from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from sparsejax import (
    SparseMatrix,
    spsolve,
)


from setup import (
    AVAILABLE_SOLVE_BACKENDS,
    ATOL,
    _device_for,
    _make_spd,
    _dense_to_sparse,
    _put,
)


@pytest.mark.parametrize("backend", AVAILABLE_SOLVE_BACKENDS)
class TestSpsolve:
    def test_forward(self, backend):
        device = _device_for(backend)
        A_dense = _make_spd(8)
        A = _dense_to_sparse(A_dense, device)
        b = _put(np.arange(1.0, 9.0), device)
        x = spsolve(A, b, backend=backend)
        want = np.linalg.solve(A_dense, np.asarray(b))
        np.testing.assert_allclose(np.asarray(x), want, atol=ATOL)

    def test_transpose(self, backend):
        device = _device_for(backend)
        A_dense = _make_spd(7, seed=4)  # SPD → A.T == A, but exercise the path
        A = _dense_to_sparse(A_dense, device)
        b = _put(np.linspace(-1, 1, 7), device)
        x = spsolve(A, b, backend=backend, transpose=True)
        want = np.linalg.solve(A_dense.T, np.asarray(b))
        np.testing.assert_allclose(np.asarray(x), want, atol=ATOL)

    def test_multiple_rhs(self, backend):
        if backend in ("cudss", "cudss_ffi"):
            pytest.skip(f"{backend} multi-rhs path is not yet supported")

        device = _device_for(backend)
        A_dense = _make_spd(6, seed=1)
        A = _dense_to_sparse(A_dense, device)
        rng = np.random.default_rng(5)
        B_np = rng.standard_normal((6, 3))
        B = _put(B_np, device)
        X = spsolve(A, B, backend=backend)
        want = np.linalg.solve(A_dense, B_np)
        np.testing.assert_allclose(np.asarray(X), want, atol=ATOL)

    def test_grad_matches_dense(self, backend):
        device = _device_for(backend)
        A_dense_np = _make_spd(5)
        b_np = np.array([1.0, -0.5, 0.25, 2.0, 0.75])

        rows, cols = np.nonzero(np.ones_like(A_dense_np))
        indices = np.stack([rows, cols]).astype(np.int32)
        data0 = jax.device_put(jnp.asarray(A_dense_np[rows, cols]), device)
        b = jax.device_put(jnp.asarray(b_np), device)

        def loss_sparse(data, b):
            A = SparseMatrix(data=data, indices=indices, shape=A_dense_np.shape)
            return jnp.sum(spsolve(A, b, backend=backend) ** 2)

        def loss_dense(flat, b):
            return jnp.sum(jnp.linalg.solve(flat.reshape(A_dense_np.shape), b) ** 2)

        g_sparse = jax.grad(loss_sparse, argnums=(0, 1))(data0, b)
        g_dense = jax.grad(loss_dense, argnums=(0, 1))(data0, b)
        np.testing.assert_allclose(
            np.asarray(g_sparse[0]), np.asarray(g_dense[0]), atol=ATOL
        )
        np.testing.assert_allclose(
            np.asarray(g_sparse[1]), np.asarray(g_dense[1]), atol=ATOL
        )
