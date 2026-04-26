"""sparsejax — Sparse linear algebra for JAX.

Public API:

    from sparsejax import SparseMatrix, spmv, spspmm, spsolve, cholesky_solve

The container (:class:`SparseMatrix`) is a registered pytree whose traced
leaf is the value array; indices and shape are static. All ops support
autograd via ``jax.grad`` through the values and right-hand side.

Backends are pluggable. The current default on CPU is SciPy (via
``jax.pure_callback``). SuiteSparse Cholmod is optional for SPD matrices
(install ``scikit-sparse``). GPU via NVIDIA cuDSS is planned via the
JAX FFI — see ``csrc/README.md``.
"""

from .sparse import SparseMatrix
from .ops import (
    spmv,
    spdmm,
    spspmm,
    spadd,
    spsolve,
    cholesky_solve,
    cholesky_solve_and_logdet,
    cholesky_factor,
    CholeskyFactor,
    logdet,
)
from . import backends

__all__ = [
    "SparseMatrix",
    "spmv",
    "spdmm",
    "spspmm",
    "spadd",
    "spsolve",
    "cholesky_solve",
    "cholesky_solve_and_logdet",
    "cholesky_factor",
    "CholeskyFactor",
    "logdet",
    "backends",
]

__version__ = "0.1.0"
