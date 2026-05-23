"""A dense mode that bypasses the sparse implementations and instad uses dense JAX operations. Useful for testing and benchmarking.

Use by wrapping the code you want to run in a `with dense_mode():` block. For example:
```python
import sparsejax as sj
with sj.dense_mode():
    grad = jax.grad(loss)(params)
```

"""

from __future__ import annotations

import threading
from contextlib import contextmanager

_state = threading.local()


def is_dense_mode() -> bool:
    return getattr(_state, "active", False)


@contextmanager
def dense_mode(enabled: bool = True):
    prev = getattr(_state, "active", False)
    _state.active = bool(enabled)
    try:
        yield
    finally:
        _state.active = prev
