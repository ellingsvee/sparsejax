from __future__ import annotations

import importlib

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from sparsejax import SparseMatrix, spspmm  # noqa: E402

spspmm_module = importlib.import_module("sparsejax.ops.matmul.spspmm")


def _backends():
    out = ["static", "scipy"]
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


def test_spspmm_auto_prefers_static_pattern_backend(monkeypatch):
    A = SparseMatrix.from_coo([1.0], [0], [0], (1, 1))

    monkeypatch.setattr(
        spspmm_module,
        "_resolve_backend",
        lambda *_args, **_kwargs: "cudss_ffi",
    )

    assert spspmm_module._resolve_spspmm_backend(A, "auto") == "static"
    assert spspmm_module._resolve_spspmm_backend(A, None) == "static"


def test_spspmm_static_handles_duplicate_input_entries():
    a_indices = np.asarray([[0, 0, 1], [0, 0, 1]], dtype=np.int32)
    b_indices = np.asarray([[0, 1], [1, 0]], dtype=np.int32)
    a_data = jnp.asarray([2.0, 3.0, 5.0])
    b_data = jnp.asarray([7.0, 11.0])
    A = SparseMatrix(data=a_data, indices=a_indices, shape=(2, 2))
    B = SparseMatrix(data=b_data, indices=b_indices, shape=(2, 2))

    C = spspmm(A, B, backend="static")

    A_dense = A.to_dense()
    B_dense = B.to_dense()
    np.testing.assert_allclose(np.asarray(C.to_dense()), np.asarray(A_dense @ B_dense))


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


@pytest.mark.parametrize("backend", _backends())
def test_spspmm_grad_dense_pattern(backend):
    """Gradient against a dense reference, with full ``[n, n]`` patterns."""
    rng = np.random.default_rng(11)
    n = 5
    rows, cols = np.nonzero(np.ones((n, n)))
    indices = np.stack([rows, cols]).astype(np.int32)
    a = jnp.asarray(rng.standard_normal(n * n))
    b = jnp.asarray(rng.standard_normal(n * n))

    def loss_sparse(a, b):
        A = SparseMatrix(data=a, indices=indices, shape=(n, n))
        B = SparseMatrix(data=b, indices=indices, shape=(n, n))
        C = spspmm(A, B, backend=backend)
        return jnp.sum(C.data**2)

    def loss_dense(a, b):
        Ad = a.reshape(n, n)
        Bd = b.reshape(n, n)
        return jnp.sum((Ad @ Bd) ** 2)

    g_sp = jax.grad(loss_sparse, argnums=(0, 1))(a, b)
    g_de = jax.grad(loss_dense, argnums=(0, 1))(a, b)
    np.testing.assert_allclose(np.asarray(g_sp[0]), np.asarray(g_de[0]), atol=1e-10)
    np.testing.assert_allclose(np.asarray(g_sp[1]), np.asarray(g_de[1]), atol=1e-10)


@pytest.mark.parametrize("backend", _backends())
def test_spspmm_grad_sparse_pattern(backend):
    """Gradient against a dense reference, where A and B have genuine
    sparsity. The bwd must project the cotangent onto each input's
    forward pattern (entries outside P_A or P_B must get zero gradient)."""
    rng = np.random.default_rng(13)
    m, k, n = 4, 5, 3
    A_dense = rng.standard_normal((m, k))
    A_dense[A_dense < 0.0] = 0.0
    B_dense = rng.standard_normal((k, n))
    B_dense[B_dense < 0.0] = 0.0

    a_rows, a_cols = np.nonzero(A_dense)
    b_rows, b_cols = np.nonzero(B_dense)
    a_indices = np.stack([a_rows, a_cols]).astype(np.int32)
    b_indices = np.stack([b_rows, b_cols]).astype(np.int32)
    a_data = jnp.asarray(A_dense[a_rows, a_cols])
    b_data = jnp.asarray(B_dense[b_rows, b_cols])

    def loss_sparse(a_data, b_data):
        A = SparseMatrix(data=a_data, indices=a_indices, shape=(m, k))
        B = SparseMatrix(data=b_data, indices=b_indices, shape=(k, n))
        C = spspmm(A, B, backend=backend)
        return jnp.sum(C.data**2)

    def loss_dense(a_data, b_data):
        Ad = jnp.zeros((m, k)).at[a_rows, a_cols].add(a_data)
        Bd = jnp.zeros((k, n)).at[b_rows, b_cols].add(b_data)
        return jnp.sum((Ad @ Bd) ** 2)

    g_sp = jax.grad(loss_sparse, argnums=(0, 1))(a_data, b_data)
    g_de = jax.grad(loss_dense, argnums=(0, 1))(a_data, b_data)
    np.testing.assert_allclose(np.asarray(g_sp[0]), np.asarray(g_de[0]), atol=1e-10)
    np.testing.assert_allclose(np.asarray(g_sp[1]), np.asarray(g_de[1]), atol=1e-10)
    # Gradient must live on each input's *forward* pattern. Since the
    # data arrays only carry P_A / P_B entries, this is a structural
    # check on the bwd: the returned cotangents are nnz-shaped, not
    # dense.
    assert g_sp[0].shape == a_data.shape
    assert g_sp[1].shape == b_data.shape


@pytest.mark.parametrize("backend", _backends())
def test_spspmm_jit(backend):
    rng = np.random.default_rng(21)
    m, k, n = 5, 6, 4
    A_dense = rng.standard_normal((m, k))
    A_dense[A_dense < 0.0] = 0.0
    B_dense = rng.standard_normal((k, n))
    B_dense[B_dense < 0.0] = 0.0
    As = SparseMatrix.from_dense(A_dense)
    Bs = SparseMatrix.from_dense(B_dense)

    @jax.jit
    def fn(a_data, b_data):
        A = SparseMatrix(data=a_data, indices=As.indices, shape=As.shape)
        B = SparseMatrix(data=b_data, indices=Bs.indices, shape=Bs.shape)
        return spspmm(A, B, backend=backend).data

    c_data = fn(As.data, Bs.data)
    C = SparseMatrix(
        data=c_data,
        indices=spspmm(As, Bs, backend=backend).indices,
        shape=(m, n),
    )
    np.testing.assert_allclose(np.asarray(C.to_dense()), A_dense @ B_dense, atol=1e-10)


@pytest.mark.parametrize("backend", _backends())
def test_spspmm_grad_chain(backend):
    """A non-trivial scalar loss that chains through ``to_dense``: the
    cotangent of ``C.data`` arrives via JAX scatter-add gradient, which
    in general has a *different* sparsity pattern than the entries we
    actually multiplied — exercising the bwd's pattern projection."""
    rng = np.random.default_rng(17)
    m, k, n = 4, 5, 4
    A_dense = rng.standard_normal((m, k))
    A_dense[A_dense < 0.0] = 0.0
    B_dense = rng.standard_normal((k, n))
    B_dense[B_dense < 0.0] = 0.0
    a_rows, a_cols = np.nonzero(A_dense)
    b_rows, b_cols = np.nonzero(B_dense)
    a_indices = np.stack([a_rows, a_cols]).astype(np.int32)
    b_indices = np.stack([b_rows, b_cols]).astype(np.int32)
    a_data = jnp.asarray(A_dense[a_rows, a_cols])
    b_data = jnp.asarray(B_dense[b_rows, b_cols])
    M = jnp.asarray(rng.standard_normal((m, n)))

    def loss_sparse(a_data, b_data):
        A = SparseMatrix(data=a_data, indices=a_indices, shape=(m, k))
        B = SparseMatrix(data=b_data, indices=b_indices, shape=(k, n))
        C = spspmm(A, B, backend=backend)
        return jnp.sum(M * C.to_dense())

    def loss_dense(a_data, b_data):
        Ad = jnp.zeros((m, k)).at[a_rows, a_cols].add(a_data)
        Bd = jnp.zeros((k, n)).at[b_rows, b_cols].add(b_data)
        return jnp.sum(M * (Ad @ Bd))

    g_sp = jax.grad(loss_sparse, argnums=(0, 1))(a_data, b_data)
    g_de = jax.grad(loss_dense, argnums=(0, 1))(a_data, b_data)
    np.testing.assert_allclose(np.asarray(g_sp[0]), np.asarray(g_de[0]), atol=1e-10)
    np.testing.assert_allclose(np.asarray(g_sp[1]), np.asarray(g_de[1]), atol=1e-10)
