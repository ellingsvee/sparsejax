from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SparseMatrix:
    data: jax.Array
    indices: np.ndarray
    shape: Tuple[int, int]

    def __post_init__(self) -> None:
        if self.data.ndim != 1:
            raise ValueError(f"data must be 1D, got shape {self.data.shape}")
        if self.indices.shape[0] != 2 or self.indices.ndim != 2:
            raise ValueError(f"indices must be [2, nnz], got {self.indices.shape}")
        if self.indices.shape[1] != self.data.shape[0]:
            raise ValueError("indices and data nnz mismatch")

    @property
    def nnz(self) -> int:
        return self.data.shape[0]

    @property
    def row(self) -> np.ndarray:
        return self.indices[0]

    @property
    def col(self) -> np.ndarray:
        return self.indices[1]

    @property
    def dtype(self):
        return self.data.dtype

    @classmethod
    def from_coo(cls, data, row, col, shape: Tuple[int, int]) -> "SparseMatrix":
        row = np.asarray(row, dtype=np.int32)
        col = np.asarray(col, dtype=np.int32)
        indices = np.stack([row, col], axis=0)
        return cls(data=jnp.asarray(data), indices=indices, shape=tuple(shape))

    @classmethod
    def from_dense(cls, dense) -> "SparseMatrix":
        dense_np = np.asarray(dense)
        row, col = np.nonzero(dense_np)
        data = dense_np[row, col]
        return cls.from_coo(data, row, col, (dense_np.shape[0], dense_np.shape[1]))

    def to_dense(self) -> jax.Array:
        m, n = self.shape
        out = jnp.zeros((m, n), dtype=self.data.dtype)
        return out.at[self.row, self.col].add(self.data)

    def transpose(self) -> "SparseMatrix":
        m, n = self.shape
        indices_T = np.stack([self.indices[1], self.indices[0]], axis=0)
        return SparseMatrix(data=self.data, indices=indices_T, shape=(n, m))

    @property
    def T(self) -> "SparseMatrix":
        return self.transpose()

    def tree_flatten(self):
        children = (self.data,)
        aux = (self.indices, self.shape)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (data,) = children
        indices, shape = aux
        # Avoid re-running __post_init__ shape checks on abstract tracers
        obj = object.__new__(cls)
        object.__setattr__(obj, "data", data)
        object.__setattr__(obj, "indices", indices)
        object.__setattr__(obj, "shape", shape)
        return obj

    def __repr__(self) -> str:
        return f"SparseMatrix(shape={self.shape}, nnz={self.nnz}, dtype={self.dtype})"
