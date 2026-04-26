from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from sparsejax import SparseMatrix, SparseStructure
from sparsejax._csr import coo_to_csr


def test_sparse_structure_can_share_pattern_across_values():
    structure = SparseStructure.from_coo([0, 1], [1, 0], (2, 2))

    A = structure.with_data(jnp.array([2.0, 3.0]))
    B = structure.with_data(jnp.array([5.0, 7.0]))

    assert isinstance(A, SparseMatrix)
    assert A.structure is structure
    assert B.structure is structure
    assert A.indices is B.indices
    np.testing.assert_allclose(np.asarray(A.to_dense()), [[0.0, 2.0], [3.0, 0.0]])
    np.testing.assert_allclose(np.asarray(B.to_dense()), [[0.0, 5.0], [7.0, 0.0]])


def test_sparse_matrix_accepts_structure_in_indices_position():
    structure = SparseStructure.from_coo([0], [0], (1, 1))
    A = SparseMatrix(jnp.array([1.0]), structure)

    assert A.structure is structure
    assert A.shape == (1, 1)


def test_csr_defaults_to_int64_for_int64_indices():
    indices = np.array([[0], [2**31]], dtype=np.int64)

    csr = coo_to_csr(indices, (1, 2**31 + 1))

    assert csr.indptr.dtype == np.int64
    assert csr.col_idx.dtype == np.int64
    assert csr.col_idx[0] == 2**31


def test_csr_int32_path_rejects_large_indices():
    indices = np.array([[0], [2**31]], dtype=np.int64)

    with pytest.raises(OverflowError, match="int32"):
        coo_to_csr(indices, (1, 2**31 + 1), index_dtype=np.int32)
