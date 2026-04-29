"""A dense mode that bypasses the sparse implementations and instad uses dense JAX operations.

The idea is that this can be used for testing, benchmarking and debugging, and provides a simple way to compare. However, unsure if this should be a part of the final package or not.

To use the dense mode, simply wrap the code you want to run in a `with dense_mode():` block. For example:
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
