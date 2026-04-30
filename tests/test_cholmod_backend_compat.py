from __future__ import annotations

import sys
from types import ModuleType

import numpy as np

from sparsejax.backends import cholmod_backend


def _install_fake_cholmod(monkeypatch, **attrs):
    sksparse = ModuleType("sksparse")
    sksparse.__path__ = []
    cholmod = ModuleType("sksparse.cholmod")
    for name, value in attrs.items():
        setattr(cholmod, name, value)
    sksparse.cholmod = cholmod
    monkeypatch.setitem(sys.modules, "sksparse", sksparse)
    monkeypatch.setitem(sys.modules, "sksparse.cholmod", cholmod)
    monkeypatch.setattr(cholmod_backend, "_CHOLMOD_API", None)


def test_cholmod_api_prefers_scikit_sparse_0_5_api(monkeypatch):
    def cho_factor(A):
        return ("new", A)

    def cholesky(A):
        return ("legacy", A)

    _install_fake_cholmod(monkeypatch, cho_factor=cho_factor, cholesky=cholesky)

    api = cholmod_backend._require_cholmod()

    assert api.name == "new"
    assert api.factorize is cho_factor


def test_cholmod_api_falls_back_to_legacy_api(monkeypatch):
    def cholesky(A):
        return ("legacy", A)

    _install_fake_cholmod(monkeypatch, cholesky=cholesky)

    api = cholmod_backend._require_cholmod()

    assert api.name == "legacy"
    assert api.factorize is cholesky


def test_cholmod_matrix_orientation_matches_api():
    row = np.array([0, 1, 1], dtype=np.int32)
    col = np.array([1, 0, 1], dtype=np.int32)
    upper = row <= col
    data = np.array([10.0, 30.0])

    new_api = cholmod_backend._CholmodApi("new", lambda A: A)
    legacy_api = cholmod_backend._CholmodApi("legacy", lambda A: A)

    new_A = cholmod_backend._coo_from_upper(
        new_api, data, row, col, upper, (2, 2)
    ).toarray()
    legacy_A = cholmod_backend._coo_from_upper(
        legacy_api, data, row, col, upper, (2, 2)
    ).toarray()

    np.testing.assert_array_equal(new_A, np.array([[0.0, 10.0], [0.0, 30.0]]))
    np.testing.assert_array_equal(legacy_A, np.array([[0.0, 0.0], [10.0, 30.0]]))
