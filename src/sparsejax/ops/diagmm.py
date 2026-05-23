from __future__ import annotations

import jax.numpy as jnp

from sparsejax.sparse import SparseMatrix


def spdiagmm(d, A: SparseMatrix) -> SparseMatrix:
    d = jnp.asarray(d)
    if d.ndim != 1 or d.shape[0] != A.shape[0]:
        raise ValueError(
            f"spdiagmm: d must be 1-D of length {A.shape[0]}, got shape {d.shape}"
        )
    return SparseMatrix(data=d[A.row] * A.data, structure=A.structure)
