#include "cudss_logdet.h"
#include "cudss_solve_logdet.h"
#include "cudss_spd_solve.h"
#include "cudss_state.h"

#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"

namespace nb = nanobind;
namespace ffi = xla::ffi;

XLA_FFI_DEFINE_HANDLER_SYMBOL(CudssSolve, (CudssSpdSolveImpl<double, ffi::F64>),
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<int64_t>("matrix_type")
                                  .Attr<int64_t>("factor_token")
                                  .Attr<int64_t>("reordering_alg")
                                  .Attr<int64_t>("factorization_alg")
                                  .Attr<int64_t>("solve_alg")
                                  .Attr<int64_t>("use_matching")
                                  .Attr<int64_t>("host_nthreads")
                                  .Attr<int64_t>("use_superpanels")
                                  .Arg<ffi::Buffer<ffi::S32>>() // row_ptr
                                  .Arg<ffi::Buffer<ffi::S32>>() // col_idx
                                  .Arg<ffi::Buffer<ffi::F64>>() // data
                                  .Arg<ffi::Buffer<ffi::F64>>() // b
                                  .Ret<ffi::Buffer<ffi::F64>>() // x
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(CudssSolveF32,
                              (CudssSpdSolveImpl<float, ffi::F32>),
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<int64_t>("matrix_type")
                                  .Attr<int64_t>("factor_token")
                                  .Attr<int64_t>("reordering_alg")
                                  .Attr<int64_t>("factorization_alg")
                                  .Attr<int64_t>("solve_alg")
                                  .Attr<int64_t>("use_matching")
                                  .Attr<int64_t>("host_nthreads")
                                  .Attr<int64_t>("use_superpanels")
                                  .Arg<ffi::Buffer<ffi::S32>>() // row_ptr
                                  .Arg<ffi::Buffer<ffi::S32>>() // col_idx
                                  .Arg<ffi::Buffer<ffi::F32>>() // data
                                  .Arg<ffi::Buffer<ffi::F32>>() // b
                                  .Ret<ffi::Buffer<ffi::F32>>() // x
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(CudssLogdet, (CudssLogdetImpl<double, ffi::F64>),
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<int64_t>("matrix_type")
                                  .Attr<int64_t>("factor_token")
                                  .Attr<int64_t>("reordering_alg")
                                  .Attr<int64_t>("factorization_alg")
                                  .Attr<int64_t>("solve_alg")
                                  .Attr<int64_t>("use_matching")
                                  .Attr<int64_t>("host_nthreads")
                                  .Attr<int64_t>("use_superpanels")
                                  .Arg<ffi::Buffer<ffi::S32>>() // row_ptr
                                  .Arg<ffi::Buffer<ffi::S32>>() // col_idx
                                  .Arg<ffi::Buffer<ffi::F64>>() // data
                                  .Ret<ffi::Buffer<ffi::F64>>() // out
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(CudssLogdetF32,
                              (CudssLogdetImpl<float, ffi::F32>),
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<int64_t>("matrix_type")
                                  .Attr<int64_t>("factor_token")
                                  .Attr<int64_t>("reordering_alg")
                                  .Attr<int64_t>("factorization_alg")
                                  .Attr<int64_t>("solve_alg")
                                  .Attr<int64_t>("use_matching")
                                  .Attr<int64_t>("host_nthreads")
                                  .Attr<int64_t>("use_superpanels")
                                  .Arg<ffi::Buffer<ffi::S32>>() // row_ptr
                                  .Arg<ffi::Buffer<ffi::S32>>() // col_idx
                                  .Arg<ffi::Buffer<ffi::F32>>() // data
                                  .Ret<ffi::Buffer<ffi::F32>>() // out
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CudssSolveLogdet, (CudssSolveLogdetImpl<double, ffi::F64>),
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("matrix_type")
        .Attr<int64_t>("factor_token")
        .Attr<int64_t>("reordering_alg")
        .Attr<int64_t>("factorization_alg")
        .Attr<int64_t>("solve_alg")
        .Attr<int64_t>("use_matching")
        .Attr<int64_t>("host_nthreads")
        .Attr<int64_t>("use_superpanels")
        .Arg<ffi::Buffer<ffi::S32>>() // row_ptr
        .Arg<ffi::Buffer<ffi::S32>>() // col_idx
        .Arg<ffi::Buffer<ffi::F64>>() // data
        .Arg<ffi::Buffer<ffi::F64>>() // b
        .Ret<ffi::Buffer<ffi::F64>>() // x
        .Ret<ffi::Buffer<ffi::F64>>() // logdet scalar
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CudssSolveLogdetF32, (CudssSolveLogdetImpl<float, ffi::F32>),
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("matrix_type")
        .Attr<int64_t>("factor_token")
        .Attr<int64_t>("reordering_alg")
        .Attr<int64_t>("factorization_alg")
        .Attr<int64_t>("solve_alg")
        .Attr<int64_t>("use_matching")
        .Attr<int64_t>("host_nthreads")
        .Attr<int64_t>("use_superpanels")
        .Arg<ffi::Buffer<ffi::S32>>() // row_ptr
        .Arg<ffi::Buffer<ffi::S32>>() // col_idx
        .Arg<ffi::Buffer<ffi::F32>>() // data
        .Arg<ffi::Buffer<ffi::F32>>() // b
        .Ret<ffi::Buffer<ffi::F32>>() // x
        .Ret<ffi::Buffer<ffi::F32>>() // logdet scalar
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
    d["cudss_solve_f32"] = EncapsulateFfiHandler(CudssSolveF32);
    d["cudss_logdet"] = EncapsulateFfiHandler(CudssLogdet);
    d["cudss_logdet_f32"] = EncapsulateFfiHandler(CudssLogdetF32);
    d["cudss_solve_logdet"] = EncapsulateFfiHandler(CudssSolveLogdet);
    d["cudss_solve_logdet_f32"] =
        EncapsulateFfiHandler(CudssSolveLogdetF32);
    return d;
  });
  m.def(
      "cudss_drop_token",
      [](int64_t token) { sparsejax::CudssCacheDrop(token); },
      "Free cached cuDSS state associated with the given factor token.");
}
