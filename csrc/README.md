# Native FFI kernels for sparsejax

This directory holds C/C++ sources that implement high-performance sparse kernels and register them as **JAX FFI targets**. They replace the scipy `pure_callback` fallbacks used by default.

## JAX FFI basics

Each kernel is a function with the `XLA_FFI_Handler` signature. It is registered with a string name and called from Python via `jax.ffi.ffi_call("name", out_shape_dtypes)(*args)`.

Minimal skeleton (`cholmod_ffi.cc`):

```cpp
#include "xla/ffi/api/ffi.h"
namespace ffi = xla::ffi;

ffi::Error CholmodSolveImpl(
    ffi::Buffer<ffi::F64> data,
    ffi::Buffer<ffi::S32> row,
    ffi::Buffer<ffi::S32> col,
    ffi::Buffer<ffi::F64> b,
    int64_t n, int64_t nnz,
    ffi::Result<ffi::Buffer<ffi::F64>> out) {
  // ... build cholmod_sparse, factorize, solve, copy into out ...
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CholmodSolve, CholmodSolveImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F64>>()  // data
        .Arg<ffi::Buffer<ffi::S32>>()  // row
        .Arg<ffi::Buffer<ffi::S32>>()  // col
        .Arg<ffi::Buffer<ffi::F64>>()  // b
        .Attr<int64_t>("n")
        .Attr<int64_t>("nnz")
        .Ret<ffi::Buffer<ffi::F64>>());
```

Python-side registration (in `backends/cholmod_backend.py`):

```python
import jax
from sparsejax._ffi import cholmod_solve_capsule  # built by setup.py

jax.ffi.register_ffi_target(
    "cholmod_solve", cholmod_solve_capsule, platform="cpu"
)
```

Then replace `pure_callback` with:

```python
out_type = jax.ShapeDtypeStruct(b.shape, b.dtype)
return jax.ffi.ffi_call(
    "cholmod_solve", out_type, vmap_method="sequential"
)(data, row_arr, col_arr, b, n=shape[0], nnz=data.shape[0])
```

## Build

The intended build is via `nanobind` + `scikit-build-core`, following the pattern in the JAX FFI docs (<https://docs.jax.dev/en/latest/ffi.html>). The extension module name is `sparsejax._ffi`; the Python package discovers it lazily so that sparsejax is usable even when the extension is not compiled.
The easiest way to build is via the `uv build` command.

## cuDSS specifics

- Include `<cudss.h>` and link against `-lcudss`.
- The handler takes GPU device pointers (from `ffi::Buffer` on the "CUDA"
  platform) and a `cudaStream_t` (via `ffi::Ctx<ffi::PlatformStream<cudaStream_t>>`).
- Reuse the cuDSS handle across calls — cache it in a static/thread-local
  to avoid reinitialization per solve.
- Register with `platform="CUDA"`.
