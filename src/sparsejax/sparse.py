from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np


_SUPPORTED_INDEX_DTYPES = (np.dtype(np.int32), np.dtype(np.int64))


def _normalize_index_dtype(index_dtype) -> np.dtype:
    dt = np.dtype(index_dtype)
    if dt not in _SUPPORTED_INDEX_DTYPES:
        raise TypeError(f"index_dtype must be int32 or int64, got {dt}")
    return dt


def _default_index_dtype(row: np.ndarray, col: np.ndarray, shape: Tuple[int, int]):
    max_i32 = np.iinfo(np.int32).max
    if max(shape) > max_i32:
        return np.dtype(np.int64)
    if row.size and (row.max(initial=0) > max_i32 or col.max(initial=0) > max_i32):
        return np.dtype(np.int64)
    return np.dtype(np.int32)


@dataclass(frozen=True)
class SparseStructure:
    """Static sparsity pattern shared by one or more value arrays."""

    indices: np.ndarray
    shape: Tuple[int, int]

    def __post_init__(self) -> None:
        indices = np.asarray(self.indices)
        if indices.ndim != 2 or indices.shape[0] != 2:
            raise ValueError(f"indices must be [2, nnz], got {indices.shape}")
        if indices.dtype not in _SUPPORTED_INDEX_DTYPES:
            indices = indices.astype(
                _default_index_dtype(indices[0], indices[1], self.shape)
            )
        shape = tuple(int(d) for d in self.shape)
        if len(shape) != 2:
            raise ValueError(f"shape must have length 2, got {self.shape}")
        if shape[0] < 0 or shape[1] < 0:
            raise ValueError(f"shape dimensions must be non-negative, got {shape}")
        if indices.size:
            row, col = indices
            if row.min(initial=0) < 0 or col.min(initial=0) < 0:
                raise ValueError("indices must be non-negative")
            if row.max(initial=-1) >= shape[0] or col.max(initial=-1) >= shape[1]:
                raise ValueError(f"indices out of bounds for shape {shape}")
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "shape", shape)

    def __hash__(self) -> int:
        return hash((id(self.indices), self.shape, self.indices.dtype.str))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, SparseStructure)
            and other.indices is self.indices
            and other.shape == self.shape
            and other.indices.dtype == self.indices.dtype
        )

    @property
    def nnz(self) -> int:
        return self.indices.shape[1]

    @property
    def row(self) -> np.ndarray:
        return self.indices[0]

    @property
    def col(self) -> np.ndarray:
        return self.indices[1]

    @property
    def index_dtype(self) -> np.dtype:
        return self.indices.dtype

    @classmethod
    def from_coo(
        cls,
        row,
        col,
        shape: Tuple[int, int],
        *,
        index_dtype=None,
    ) -> "SparseStructure":
        row_arr = np.asarray(row)
        col_arr = np.asarray(col)
        if row_arr.shape != col_arr.shape:
            raise ValueError(
                "row and col shapes must match, "
                f"got {row_arr.shape} and {col_arr.shape}"
            )
        if index_dtype is None:
            index_dtype = _default_index_dtype(row_arr, col_arr, tuple(shape))
        dt = _normalize_index_dtype(index_dtype)
        indices = np.stack(
            [row_arr.astype(dt, copy=False), col_arr.astype(dt, copy=False)], axis=0
        )
        return cls(indices=indices, shape=tuple(shape))

    def transpose(self) -> "SparseStructure":
        m, n = self.shape
        indices_t = np.stack([self.indices[1], self.indices[0]], axis=0)
        return SparseStructure(indices=indices_t, shape=(n, m))

    @property
    def T(self) -> "SparseStructure":
        return self.transpose()

    def csr(self, *, index_dtype=None):
        from sparsejax._csr import coo_to_csr

        return coo_to_csr(self.indices, self.shape, index_dtype=index_dtype)

    def with_data(self, data) -> "SparseMatrix":
        return SparseMatrix(data=data, structure=self)


@jax.tree_util.register_pytree_node_class
class SparseMatrix:
    __slots__ = ("data", "structure")

    data: jax.Array
    structure: SparseStructure

    def __init__(
        self,
        data,
        indices: np.ndarray | SparseStructure | None = None,
        shape: Tuple[int, int] | None = None,
        *,
        structure: SparseStructure | None = None,
    ) -> None:
        if structure is None:
            if isinstance(indices, SparseStructure):
                if shape is not None:
                    raise ValueError(
                        "shape must be omitted when indices is a SparseStructure"
                    )
                structure = indices
            else:
                if indices is None or shape is None:
                    raise ValueError(
                        "SparseMatrix requires indices and shape, or structure"
                    )
                structure = SparseStructure(
                    indices=np.asarray(indices), shape=tuple(shape)
                )
        elif indices is not None or shape is not None:
            raise ValueError("pass either structure or indices/shape, not both")

        if data.ndim != 1:
            raise ValueError(f"data must be 1D, got shape {data.shape}")
        if structure.nnz != data.shape[0]:
            raise ValueError("indices and data nnz mismatch")
        self.data = data
        self.structure = structure

    @property
    def nnz(self) -> int:
        return self.structure.nnz

    @property
    def indices(self) -> np.ndarray:
        return self.structure.indices

    @property
    def shape(self) -> Tuple[int, int]:
        return self.structure.shape

    @property
    def row(self) -> np.ndarray:
        return self.structure.row

    @property
    def col(self) -> np.ndarray:
        return self.structure.col

    @property
    def dtype(self):
        return self.data.dtype

    @classmethod
    def from_coo(
        cls,
        data,
        row,
        col,
        shape: Tuple[int, int],
        *,
        index_dtype=None,
        sum_duplicates: bool = False,
    ) -> "SparseMatrix":
        """Build a SparseMatrix from COO triplets.

        With ``sum_duplicates=True``, repeated ``(row, col)`` entries are deduplicated statically (numpy) and their values summed inside XLA.
        """
        if not sum_duplicates:
            structure = SparseStructure.from_coo(
                row, col, shape, index_dtype=index_dtype
            )
            return cls(data=jnp.asarray(data), structure=structure)

        rows_np = np.asarray(row, dtype=np.int64)
        cols_np = np.asarray(col, dtype=np.int64)
        if rows_np.shape != cols_np.shape:
            raise ValueError(
                "row and col shapes must match, "
                f"got {rows_np.shape} and {cols_np.shape}"
            )
        data_arr = jnp.asarray(data)
        if data_arr.shape != rows_np.shape:
            raise ValueError(
                "data and row shapes must match, "
                f"got {data_arr.shape} and {rows_np.shape}"
            )
        n_cols = int(shape[1])
        lin = rows_np * n_cols + cols_np
        uniq, inv = np.unique(lin, return_inverse=True)
        nnz = int(uniq.size)
        out_rows = (uniq // n_cols).astype(np.int64)
        out_cols = (uniq % n_cols).astype(np.int64)
        structure = SparseStructure.from_coo(
            out_rows, out_cols, shape, index_dtype=index_dtype
        )
        out = jnp.zeros(nnz, dtype=data_arr.dtype).at[jnp.asarray(inv)].add(data_arr)
        return cls(data=out, structure=structure)

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
        return SparseMatrix(data=self.data, structure=self.structure.T)

    @property
    def T(self) -> "SparseMatrix":
        return self.transpose()

    def tree_flatten(self):
        children = (self.data,)
        aux = (self.structure,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (data,) = children
        (structure,) = aux
        # Avoid re-running __post_init__ shape checks on abstract tracers
        obj = object.__new__(cls)
        object.__setattr__(obj, "data", data)
        object.__setattr__(obj, "structure", structure)
        return obj

    def __repr__(self) -> str:
        return f"SparseMatrix(shape={self.shape}, nnz={self.nnz}, dtype={self.dtype})"

    def __mul__(self, other) -> "SparseMatrix":
        if isinstance(other, (int, float, jax.Array, jnp.ndarray)):
            return SparseMatrix(self.data * other, structure=self.structure)
        return NotImplemented

    def __rmul__(self, other) -> "SparseMatrix":
        return self.__mul__(other)

    def __truediv__(self, other) -> "SparseMatrix":
        if isinstance(other, (int, float, jax.Array, jnp.ndarray)):
            return SparseMatrix(self.data / other, structure=self.structure)
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
        return SparseMatrix(-self.data, structure=self.structure)

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
