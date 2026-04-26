"""Symmetric part of a sparse matrix."""

from __future__ import annotations

from sparsejax.sparse import SparseMatrix
from sparsejax.ops.add import spadd


def symmetrize(A: SparseMatrix) -> SparseMatrix:
    """Return ``0.5 * (A + A.T)``."""
    return 0.5 * spadd(A, A.T)
