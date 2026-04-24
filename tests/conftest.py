"""Test-level setup shared across test modules.

Enables x64 before tests import JAX-using modules, and exposes a set of
helpers (setup.py lives next to it for backward-compat).
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

# Allow `from setup import ...` — tests were written that way.
sys.path.insert(0, str(Path(__file__).parent))
