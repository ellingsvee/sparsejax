"""Sparse matrix constructors for common patterns (identity, diagonal)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from sparsejax.sparse import SparseMatrix, SparseStructure


def eye(n: int, dtype=jnp.float64) -> SparseMatrix:
    """Sparse (n,n) identity matrix."""
    n = int(n)
    idx = np.arange(n, dtype=np.int32)
    indices = np.stack([idx, idx], axis=0)
    structure = SparseStructure(indices=indices, shape=(n, n))
    return SparseMatrix(data=jnp.ones(n, dtype=dtype), structure=structure)


def diag(d) -> SparseMatrix:
    """Sparse diagonal matrix with values from array d on the diagonal."""
    d = jnp.asarray(d)
    if d.ndim != 1:
        raise ValueError(f"diag: d must be 1-D, got shape {d.shape}")
    n = int(d.shape[0])
    idx = np.arange(n, dtype=np.int32)
    indices = np.stack([idx, idx], axis=0)
    structure = SparseStructure(indices=indices, shape=(n, n))
    return SparseMatrix(data=d, structure=structure)
