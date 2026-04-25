"""COO → CSR conversion + per-pattern factor token issuance.

A factor token is a stable, process-unique int64 that identifies a sparsity
pattern. Native FFI backends (cuDSS) use it as a cache key for the cuDSS
analysis phase, so repeated calls with the same indices array — even with
different numerical values — reuse the symbolic factorization.

Tokens are tied to the lifetime of the indices array via a weakref
finalizer that calls back into the native module to free the cached cuDSS
state. The mapping from `id(indices)` → token is held in a regular dict.
The cached `CsrStructure` stores a reference to the indices object so a
lookup that hits a stale entry from id reuse (possible if the original
indices array wasn't weakref-able and finalize never fired) is detected
via an identity check and rebuilt.
"""

from __future__ import annotations

from typing import NamedTuple, Tuple
import threading
import weakref

import numpy as np


class CsrStructure(NamedTuple):
    indptr: np.ndarray  # int32, shape (n_rows + 1,)
    col_idx: np.ndarray  # int32, shape (nnz,)
    order: np.ndarray  # int64, data-permutation (data_csr = data[order])
    shape: Tuple[int, int]
    factor_token: int  # nonzero, stable per indices array
    # Weak ref to the indices ndarray this structure was built for, used to
    # detect id reuse on cache lookup. Strong ref would pin `indices` for
    # the process lifetime and defeat the weakref-finalize cleanup.
    indices_ref: "weakref.ReferenceType[np.ndarray] | None"


# Map id(indices) -> CsrStructure. The cache holds a strong ref to the
# CsrStructure (the previous WeakValueDictionary design dropped it
# instantly because nothing else held one, which silently invalidated the
# factor_token across calls). Lifetime is tied to the indices array via
# weakref.finalize, which removes the entry — and frees the cuDSS cached
# state — when indices is GC'd.
_CACHE: dict[int, CsrStructure] = {}
_TOKEN_LOCK = threading.Lock()
_NEXT_TOKEN = 1


def _alloc_token() -> int:
    global _NEXT_TOKEN
    with _TOKEN_LOCK:
        tok = _NEXT_TOKEN
        _NEXT_TOKEN += 1
    return tok


def _native_drop_token(token: int) -> None:
    """Best-effort free of the cached cuDSS state for `token`."""
    try:
        from sparsejax import backend as _native  # type: ignore
    except ImportError:
        return
    drop = getattr(_native, "cudss_drop_token", None)
    if drop is not None:
        try:
            drop(int(token))
        except Exception:
            pass


def _on_indices_collected(key: int, token: int) -> None:
    _CACHE.pop(key, None)
    _native_drop_token(token)


def coo_to_csr(indices: np.ndarray, shape: Tuple[int, int]) -> CsrStructure:
    key = id(indices)
    entry = _CACHE.get(key)
    if entry is not None and entry.shape == shape:
        # Reject stale entries left behind by id reuse: if the cached
        # weakref still resolves to a different object, or has expired
        # while a new ndarray now occupies the same id, we treat it as a
        # miss. (`indices_ref is None` means the original wasn't
        # weakref-able; we can't verify identity, so we trust the id and
        # accept the small id-reuse risk for that path.)
        cached = entry.indices_ref() if entry.indices_ref is not None else None
        if entry.indices_ref is None or cached is indices:
            return entry
    if entry is not None:
        # Stale entry — drop the cached cuDSS state too before rebuilding.
        _native_drop_token(entry.factor_token)
        _CACHE.pop(key, None)
    row = np.asarray(indices[0], dtype=np.int64)
    col = np.asarray(indices[1], dtype=np.int64)
    n_rows = shape[0]
    # Sort primarily by row, secondarily by col — gives canonical CSR layout.
    order = np.lexsort((col, row))
    row_sorted = row[order]
    col_sorted = col[order]
    counts = np.bincount(row_sorted, minlength=n_rows)
    indptr = np.empty(n_rows + 1, dtype=np.int32)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    token = _alloc_token()
    try:
        indices_ref: "weakref.ReferenceType[np.ndarray] | None" = weakref.ref(indices)
    except TypeError:
        indices_ref = None
    csr = CsrStructure(
        indptr=indptr,
        col_idx=col_sorted.astype(np.int32, copy=False),
        order=order,
        shape=shape,
        factor_token=token,
        indices_ref=indices_ref,
    )
    _CACHE[key] = csr
    try:
        weakref.finalize(indices, _on_indices_collected, key, token)
    except TypeError:
        # Object isn't weakref-able — fall back to leaking the entry.
        pass
    return csr
