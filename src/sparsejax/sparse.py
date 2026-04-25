from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np


class _HashableIndices:
    """Wraps the indices ndarray so it can sit in a JIT cache key."""

    __slots__ = ("array",)

    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def __hash__(self) -> int:
        return id(self.array)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _HashableIndices) and other.array is self.array


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
        aux = (_HashableIndices(self.indices), self.shape)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (data,) = children
        indices_holder, shape = aux
        indices = (
            indices_holder.array
            if isinstance(indices_holder, _HashableIndices)
            else indices_holder
        )
        # Avoid re-running __post_init__ shape checks on abstract tracers
        obj = object.__new__(cls)
        object.__setattr__(obj, "data", data)
        object.__setattr__(obj, "indices", indices)
        object.__setattr__(obj, "shape", shape)
        return obj

    def __repr__(self) -> str:
        return f"SparseMatrix(shape={self.shape}, nnz={self.nnz}, dtype={self.dtype})"

    def __mul__(self, other) -> "SparseMatrix":
        if isinstance(other, (int, float, jax.Array, jnp.ndarray)):
            return SparseMatrix(self.data * other, self.indices, self.shape)
        return NotImplemented

    def __rmul__(self, other) -> "SparseMatrix":
        return self.__mul__(other)

    def __truediv__(self, other) -> "SparseMatrix":
        if isinstance(other, (int, float, jax.Array, jnp.ndarray)):
            return SparseMatrix(self.data / other, self.indices, self.shape)
        return NotImplemented

    def __add__(self, other) -> jax.Array | "SparseMatrix":
        # If adding a scalar, the result is dense
        if (
            isinstance(other, (int, float, jax.Array, jnp.ndarray))
            and jnp.ndim(other) == 0
        ):
            return self.to_dense() + other

        # If adding another SparseMatrix
        if isinstance(other, SparseMatrix):
            from sparsejax.ops.add import spadd

            return spadd(self, other)

        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self) -> "SparseMatrix":
        return SparseMatrix(-self.data, self.indices, self.shape)

    def __matmul__(self, other) -> jax.Array | "SparseMatrix":
        if isinstance(other, (jax.Array, jnp.ndarray)):
            from sparsejax.ops.matmul import spdmm

            return spdmm(self, other)

        if isinstance(other, SparseMatrix):
            from sparsejax.ops.matmul import spspmm

            return spspmm(self, other)

        return NotImplemented

    def __rmatmul__(self, other):
        return NotImplemented
