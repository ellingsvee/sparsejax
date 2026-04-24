#include "cudss_logdet.h"
#include "cudss_spd_solve.h"
#include "cudss_state.h"

#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"

namespace nb = nanobind;
namespace ffi = xla::ffi;

XLA_FFI_DEFINE_HANDLER_SYMBOL(CudssSolve, CudssSpdSolveImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<int64_t>("matrix_type")
                                  .Attr<int64_t>("factor_token")
                                  .Arg<ffi::Buffer<ffi::S32>>() // row_ptr
                                  .Arg<ffi::Buffer<ffi::S32>>() // col_idx
                                  .Arg<ffi::Buffer<ffi::F64>>() // data
                                  .Arg<ffi::Buffer<ffi::F64>>() // b
                                  .Ret<ffi::Buffer<ffi::F64>>() // x
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(CudssLogdet, CudssLogdetImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<int64_t>("matrix_type")
                                  .Attr<int64_t>("factor_token")
                                  .Arg<ffi::Buffer<ffi::S32>>() // row_ptr
                                  .Arg<ffi::Buffer<ffi::S32>>() // col_idx
                                  .Arg<ffi::Buffer<ffi::F64>>() // data
                                  .Ret<ffi::Buffer<ffi::F64>>() // out (scalar)
);

template <typename T> nb::capsule EncapsulateFfiHandler(T *fn) {
  static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                "Encapsulated function must be an XLA FFI handler");
  return nb::capsule(reinterpret_cast<void *>(fn));
}

NB_MODULE(backend, m) {
  m.def("registrations", []() {
    nb::dict d;
    d["cudss_solve"] = EncapsulateFfiHandler(CudssSolve);
    d["cudss_logdet"] = EncapsulateFfiHandler(CudssLogdet);
    return d;
  });
  m.def(
      "cudss_drop_token",
      [](int64_t token) { sparsejax::CudssCacheDrop(token); },
      "Free cached cuDSS state associated with the given factor token.");
}
