from __future__ import annotations


import jax

from sparsejax.sparse import SparseMatrix
from sparsejax import backends


def _normalize_device_kind(device_kind: str) -> str:
    if device_kind in ("cuda", "rocm"):
        return "gpu"
    return device_kind


def _device_kind_for_data(data) -> str:
    try:
        device = data.device
        if callable(device):
            device = device()
        platform = getattr(device, "platform", None)
        if platform is not None:
            return _normalize_device_kind(platform)
    except Exception:
        pass

    # Inside jit/grad, ``data`` may be a tracer without a concrete device.
    # Prefer JAX's active default backend instead of silently falling back to CPU.
    try:
        return _normalize_device_kind(jax.default_backend())
    except Exception:
        return "cpu"


def _resolve_backend(A: SparseMatrix, backend: str | None, spd: bool = True) -> str:
    if backend is not None and backend != "auto":
        return backend
    device_kind = _device_kind_for_data(A.data)
    return backends.select_backend(device_kind, spd=spd)
