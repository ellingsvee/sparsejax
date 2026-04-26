from __future__ import annotations

import jax

from sparsejax import backends
from sparsejax.utils import _device_kind_for_data, _resolve_backend


class _TracerLikeData:
    pass


class _FakeMatrix:
    data = _TracerLikeData()


def test_device_kind_for_tracer_like_data_uses_default_backend(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")

    assert _device_kind_for_data(_TracerLikeData()) == "gpu"


def test_resolve_backend_uses_default_backend_for_tracer_like_data(monkeypatch):
    calls = []

    def fake_select_backend(device_kind: str, *, spd: bool = False) -> str:
        calls.append((device_kind, spd))
        return "cudss_ffi"

    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(backends, "select_backend", fake_select_backend)

    assert _resolve_backend(_FakeMatrix(), "auto", spd=True) == "cudss_ffi"
    assert calls == [("gpu", True)]
