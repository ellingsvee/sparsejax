from __future__ import annotations

from typing import Dict, Literal

BackendName = Literal["scipy", "cholmod", "cudss", "cudss_ffi"]

_AVAILABILITY: Dict[str, bool] = {}


def _probe(name: str) -> bool:
    if name in _AVAILABILITY:
        return _AVAILABILITY[name]
    try:
        if name == "scipy":
            import scipy.sparse  # noqa: F401

            ok = True
        elif name == "cholmod":
            from sksparse import cholmod  # noqa: F401

            ok = True
        elif name == "cudss":
            import cupy  # noqa: F401
            from nvmath.sparse.advanced import direct_solver  # noqa: F401

            ok = True
        elif name == "cudss_ffi":
            from . import cudss_ffi_backend

            ok = cudss_ffi_backend.is_available()
        else:
            ok = False
    except Exception:
        ok = False
    _AVAILABILITY[name] = ok
    return ok


def is_available(name: BackendName) -> bool:
    return _probe(name)


def available_backends() -> list[str]:
    return [n for n in ("scipy", "cholmod", "cudss_ffi", "cudss") if _probe(n)]


def select_backend(device_kind: str, *, spd: bool = False) -> str:
    """Pick a reasonable default backend.

    Parameters
    ----------
    device_kind : str
        "cpu" or "gpu" (as reported by ``jax.Array.device().platform``).
    spd : bool
        Whether the matrix is known SPD — if so, prefer Cholmod on CPU.
    """
    if device_kind == "gpu" and is_available("cudss_ffi"):
        return "cudss_ffi"
    if device_kind == "gpu" and is_available("cudss"):
        return "cudss"
    if spd and is_available("cholmod"):
        return "cholmod"
    if is_available("scipy"):
        return "scipy"
    raise RuntimeError(
        "No sparse backend available. Install scipy (`pip install scipy`) "
        "for the baseline CPU backend."
    )
