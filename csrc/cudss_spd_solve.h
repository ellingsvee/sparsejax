// cuDSS-backed sparse direct solver exposed as a JAX FFI handler.

#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <type_traits>

#include "cudss.h"
#include "cudss_state.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

#ifndef SPARSEJAX_CUDSS_CHECK
#define SPARSEJAX_CUDSS_CHECK(expr)                                            \
  do {                                                                         \
    cudssStatus_t _s = (expr);                                                 \
    if (_s != CUDSS_STATUS_SUCCESS) {                                          \
      return ffi::Error::Internal(                                             \
          std::string("cuDSS call failed: " #expr " -> status=") +             \
          std::to_string(static_cast<int>(_s)));                               \
    }                                                                          \
  } while (0)
#endif

#ifndef SPARSEJAX_CUDA_CHECK
#define SPARSEJAX_CUDA_CHECK(expr)                                             \
  do {                                                                         \
    cudaError_t _e = (expr);                                                   \
    if (_e != cudaSuccess) {                                                   \
      return ffi::Error::Internal(                                             \
          std::string("CUDA call failed: " #expr " -> ") +                     \
          cudaGetErrorString(_e));                                             \
    }                                                                          \
  } while (0)
#endif

namespace sparsejax {

inline std::string CudssStatusString(cudssStatus_t s) {
  switch (s) {
  case CUDSS_STATUS_SUCCESS:
    return "CUDSS_STATUS_SUCCESS";
  case CUDSS_STATUS_NOT_INITIALIZED:
    return "CUDSS_STATUS_NOT_INITIALIZED";
  case CUDSS_STATUS_ALLOC_FAILED:
    return "CUDSS_STATUS_ALLOC_FAILED";
  case CUDSS_STATUS_INVALID_VALUE:
    return "CUDSS_STATUS_INVALID_VALUE";
  case CUDSS_STATUS_NOT_SUPPORTED:
    return "CUDSS_STATUS_NOT_SUPPORTED";
  case CUDSS_STATUS_EXECUTION_FAILED:
    return "CUDSS_STATUS_EXECUTION_FAILED";
  case CUDSS_STATUS_INTERNAL_ERROR:
    return "CUDSS_STATUS_INTERNAL_ERROR";
  default:
    return "CUDSS_STATUS_UNKNOWN";
  }
}

inline ffi::Error CudssError(const char *what, cudssStatus_t s) {
  return ffi::Error::Internal(std::string(what) + " failed: " +
                              CudssStatusString(s) + " (" +
                              std::to_string(static_cast<int>(s)) + ")");
}

template <typename T> constexpr cudaDataType_t CudaValueType();
template <> constexpr cudaDataType_t CudaValueType<float>() {
  return CUDA_R_32F;
}
template <> constexpr cudaDataType_t CudaValueType<double>() {
  return CUDA_R_64F;
}

inline ffi::Error EncodeMatrixType(int64_t matrix_type,
                                   cudssMatrixType_t &mtype,
                                   cudssMatrixViewType_t &mview) {
  switch (matrix_type) {
  case 1:
    mtype = CUDSS_MTYPE_SPD;
    mview = CUDSS_MVIEW_UPPER;
    break;
  case 2:
    mtype = CUDSS_MTYPE_SYMMETRIC;
    mview = CUDSS_MVIEW_UPPER;
    break;
  case 0:
  default:
    mtype = CUDSS_MTYPE_GENERAL;
    mview = CUDSS_MVIEW_FULL;
    break;
  }
  return ffi::Error::Success();
}

// Ensure `entry` has a live (handle, config, data) bound to `stream`. If the
// cached pattern dimensions or matrix type changed, the entry is reset and a
// fresh state is created so analysis is rerun.
inline ffi::Error EnsureCudssState(CudssEntry &entry, cudaStream_t stream,
                                   int64_t n, int64_t nnz, int matrix_type,
                                   int reordering_alg, int factorization_alg,
                                   int solve_alg, int use_matching,
                                   int host_nthreads, int use_superpanels) {
  if (entry.analyzed &&
      (entry.n != n || entry.nnz != nnz || entry.matrix_type != matrix_type ||
       entry.reordering_alg != reordering_alg ||
       entry.factorization_alg != factorization_alg ||
       entry.solve_alg != solve_alg || entry.use_matching != use_matching ||
       entry.host_nthreads != host_nthreads ||
       entry.use_superpanels != use_superpanels)) {
    if (entry.data && entry.handle)
      cudssDataDestroy(entry.handle, entry.data);
    if (entry.config)
      cudssConfigDestroy(entry.config);
    if (entry.handle)
      cudssDestroy(entry.handle);
    entry.handle = nullptr;
    entry.config = nullptr;
    entry.data = nullptr;
    entry.analyzed = false;
  }
  if (!entry.handle) {
    SPARSEJAX_CUDSS_CHECK(cudssCreate(&entry.handle));
  }
  SPARSEJAX_CUDSS_CHECK(cudssSetStream(entry.handle, stream));
  if (!entry.config) {
    SPARSEJAX_CUDSS_CHECK(cudssConfigCreate(&entry.config));
  }
  auto set_config = [&](cudssConfigParam_t param, int value,
                        const char *name) -> ffi::Error {
    if (value < 0) {
      return ffi::Error::Success();
    }
    cudssStatus_t s = cudssConfigSet(entry.config, param, &value, sizeof(value));
    if (s != CUDSS_STATUS_SUCCESS) {
      return CudssError(name, s);
    }
    return ffi::Error::Success();
  };
  if (auto err = set_config(CUDSS_CONFIG_REORDERING_ALG, reordering_alg,
                            "cudssConfigSet(REORDERING_ALG)");
      err.failure())
    return err;
  if (auto err = set_config(CUDSS_CONFIG_FACTORIZATION_ALG, factorization_alg,
                            "cudssConfigSet(FACTORIZATION_ALG)");
      err.failure())
    return err;
  if (auto err =
          set_config(CUDSS_CONFIG_SOLVE_ALG, solve_alg, "cudssConfigSet(SOLVE_ALG)");
      err.failure())
    return err;
  if (auto err = set_config(CUDSS_CONFIG_USE_MATCHING, use_matching,
                            "cudssConfigSet(USE_MATCHING)");
      err.failure())
    return err;
  if (auto err = set_config(CUDSS_CONFIG_HOST_NTHREADS, host_nthreads,
                            "cudssConfigSet(HOST_NTHREADS)");
      err.failure())
    return err;
  if (auto err = set_config(CUDSS_CONFIG_USE_SUPERPANELS, use_superpanels,
                            "cudssConfigSet(USE_SUPERPANELS)");
      err.failure())
    return err;
  if (!entry.data) {
    SPARSEJAX_CUDSS_CHECK(cudssDataCreate(entry.handle, &entry.data));
  }
  entry.n = n;
  entry.nnz = nnz;
  entry.matrix_type = matrix_type;
  entry.reordering_alg = reordering_alg;
  entry.factorization_alg = factorization_alg;
  entry.solve_alg = solve_alg;
  entry.use_matching = use_matching;
  entry.host_nthreads = host_nthreads;
  entry.use_superpanels = use_superpanels;
  return ffi::Error::Success();
}

} // namespace sparsejax

template <typename T, ffi::DataType FfiT>
inline ffi::Error
CudssSpdSolveImpl(cudaStream_t stream, int64_t matrix_type,
                  int64_t factor_token, int64_t reordering_alg,
                  int64_t factorization_alg, int64_t solve_alg,
                  int64_t use_matching, int64_t host_nthreads,
                  int64_t use_superpanels, ffi::Buffer<ffi::S32> row_ptr,
                  ffi::Buffer<ffi::S32> col_idx, ffi::Buffer<FfiT> data,
                  ffi::Buffer<FfiT> b, ffi::ResultBuffer<FfiT> x) {
  // Layout convention: cuDSS dense matrices are COL_MAJOR with ld=n. The
  // Python wrapper requests that layout for 2-D dense FFI buffers, so shape
  // remains the public (n, nrhs) solve shape while the physical layout matches
  // cuDSS.
  const auto &b_dims = b.dimensions();
  if (b_dims.size() < 1 || b_dims.size() > 2) {
    return ffi::Error::InvalidArgument("b must be 1-D or 2-D");
  }
  const auto &rp_dims = row_ptr.dimensions();
  if (rp_dims.size() != 1 || rp_dims[0] < 1) {
    return ffi::Error::InvalidArgument("row_ptr must be 1-D with size n+1");
  }
  const int64_t n = rp_dims[0] - 1;
  int64_t nrhs;
  if (b_dims.size() == 1) {
    if (b_dims[0] != n) {
      return ffi::Error::InvalidArgument("1-D b length must equal n");
    }
    nrhs = 1;
  } else {
    if (b_dims[0] != n) {
      return ffi::Error::InvalidArgument(
          "2-D b must have shape (n, nrhs)");
    }
    nrhs = b_dims[1];
  }
  const int64_t nnz = col_idx.dimensions()[0];
  if (data.dimensions()[0] != nnz) {
    return ffi::Error::InvalidArgument("data and col_idx must agree on nnz");
  }

  cudssMatrixType_t mtype;
  cudssMatrixViewType_t mview;
  if (auto err = sparsejax::EncodeMatrixType(matrix_type, mtype, mview);
      err.failure()) {
    return err;
  }

  // factor_token == 0 -> ephemeral state (no cross-call reuse). Otherwise we
  // pull a persistent entry from the global cache, keyed by token.
  std::unique_ptr<sparsejax::CudssEntry> ephemeral;
  std::unique_lock<std::mutex> lock;
  sparsejax::CudssEntry *entry_ptr = nullptr;
  if (factor_token == 0) {
    ephemeral = std::make_unique<sparsejax::CudssEntry>();
    entry_ptr = ephemeral.get();
  } else {
    entry_ptr =
        &sparsejax::CudssCache::Instance().GetOrCreate(factor_token, lock);
  }
  sparsejax::CudssEntry &entry = *entry_ptr;

  if (auto err = sparsejax::EnsureCudssState(entry, stream, n, nnz,
                                             static_cast<int>(matrix_type),
                                             static_cast<int>(reordering_alg),
                                             static_cast<int>(factorization_alg),
                                             static_cast<int>(solve_alg),
                                             static_cast<int>(use_matching),
                                             static_cast<int>(host_nthreads),
                                             static_cast<int>(use_superpanels));
      err.failure()) {
    return err;
  }

  cudssMatrix_t A = nullptr, X = nullptr, B = nullptr;
  auto cleanup = [&]() {
    if (A)
      cudssMatrixDestroy(A);
    if (B)
      cudssMatrixDestroy(B);
    if (X)
      cudssMatrixDestroy(X);
  };

  cudssStatus_t s;
  s = cudssMatrixCreateCsr(&A, n, n, nnz, row_ptr.typed_data(), nullptr,
                           col_idx.typed_data(), data.typed_data(), CUDA_R_32I,
                           sparsejax::CudaValueType<T>(), mtype, mview,
                           CUDSS_BASE_ZERO);
  if (s != CUDSS_STATUS_SUCCESS) {
    cleanup();
    return sparsejax::CudssError("cudssMatrixCreateCsr", s);
  }

  const int64_t ld = n;
  s = cudssMatrixCreateDn(&B, n, nrhs, ld, b.typed_data(),
                          sparsejax::CudaValueType<T>(), CUDSS_LAYOUT_COL_MAJOR);
  if (s != CUDSS_STATUS_SUCCESS) {
    cleanup();
    return sparsejax::CudssError("cudssMatrixCreateDn(B)", s);
  }
  s = cudssMatrixCreateDn(&X, n, nrhs, ld, x->typed_data(),
                          sparsejax::CudaValueType<T>(), CUDSS_LAYOUT_COL_MAJOR);
  if (s != CUDSS_STATUS_SUCCESS) {
    cleanup();
    return sparsejax::CudssError("cudssMatrixCreateDn(X)", s);
  }

  if (!entry.analyzed) {
    s = cudssExecute(entry.handle, CUDSS_PHASE_ANALYSIS, entry.config,
                     entry.data, A, X, B);
    if (s != CUDSS_STATUS_SUCCESS) {
      cleanup();
      return sparsejax::CudssError("cudssExecute(ANALYSIS)", s);
    }
    s = cudssExecute(entry.handle, CUDSS_PHASE_FACTORIZATION, entry.config,
                     entry.data, A, X, B);
    if (s != CUDSS_STATUS_SUCCESS) {
      cleanup();
      return sparsejax::CudssError("cudssExecute(FACTORIZATION)", s);
    }
    entry.analyzed = true;
  } else {
    s = cudssExecute(entry.handle, CUDSS_PHASE_REFACTORIZATION, entry.config,
                     entry.data, A, X, B);
    if (s != CUDSS_STATUS_SUCCESS) {
      cleanup();
      return sparsejax::CudssError("cudssExecute(REFACTORIZATION)", s);
    }
  }

  s = cudssExecute(entry.handle, CUDSS_PHASE_SOLVE, entry.config, entry.data, A,
                   X, B);
  if (s != CUDSS_STATUS_SUCCESS) {
    cleanup();
    return sparsejax::CudssError("cudssExecute(SOLVE)", s);
  }

  cleanup();
  return ffi::Error::Success();
}
