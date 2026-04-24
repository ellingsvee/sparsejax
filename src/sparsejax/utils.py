from __future__ import annotations


from sparsejax.sparse import SparseMatrix
from sparsejax import backends


def _resolve_backend(A: SparseMatrix, backend: str | None, spd: bool = True) -> str:
    if backend is not None and backend != "auto":
        return backend
    try:
        device_kind = A.data.device.platform
    except Exception:
        device_kind = "cpu"
    return backends.select_backend(device_kind, spd=spd)
