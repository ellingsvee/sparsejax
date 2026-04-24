// cuDSS-backed sparse direct solver exposed as a JAX FFI handler.

#pragma once

#include <cstdint>
#include <mutex>
#include <string>

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
                                   int64_t n, int64_t nnz, int matrix_type) {
  if (entry.analyzed &&
      (entry.n != n || entry.nnz != nnz || entry.matrix_type != matrix_type)) {
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
  if (!entry.data) {
    SPARSEJAX_CUDSS_CHECK(cudssDataCreate(entry.handle, &entry.data));
  }
  entry.n = n;
  entry.nnz = nnz;
  entry.matrix_type = matrix_type;
  return ffi::Error::Success();
}

} // namespace sparsejax

inline ffi::Error
CudssSpdSolveImpl(cudaStream_t stream, int64_t matrix_type,
                  int64_t factor_token, ffi::Buffer<ffi::S32> row_ptr,
                  ffi::Buffer<ffi::S32> col_idx, ffi::Buffer<ffi::F64> data,
                  ffi::Buffer<ffi::F64> b, ffi::ResultBuffer<ffi::F64> x) {
  const auto &b_dims = b.dimensions();
  if (b_dims.size() < 1 || b_dims.size() > 2) {
    return ffi::Error::InvalidArgument("b must be 1-D or 2-D");
  }
  const int64_t n = b_dims[0];
  const int64_t nrhs = (b_dims.size() == 2) ? b_dims[1] : 1;

  const auto &rp_dims = row_ptr.dimensions();
  if (rp_dims.size() != 1 || rp_dims[0] != n + 1) {
    return ffi::Error::InvalidArgument("row_ptr must have shape (n+1,)");
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
                                             static_cast<int>(matrix_type));
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
                           CUDA_R_64F, mtype, mview, CUDSS_BASE_ZERO);
  if (s != CUDSS_STATUS_SUCCESS) {
    cleanup();
    return ffi::Error::Internal("cudssMatrixCreateCsr failed");
  }

  const int64_t ld = n;
  s = cudssMatrixCreateDn(&B, n, nrhs, ld, b.typed_data(), CUDA_R_64F,
                          CUDSS_LAYOUT_COL_MAJOR);
  if (s != CUDSS_STATUS_SUCCESS) {
    cleanup();
    return ffi::Error::Internal("cudssMatrixCreateDn(B) failed");
  }
  s = cudssMatrixCreateDn(&X, n, nrhs, ld, x->typed_data(), CUDA_R_64F,
                          CUDSS_LAYOUT_COL_MAJOR);
  if (s != CUDSS_STATUS_SUCCESS) {
    cleanup();
    return ffi::Error::Internal("cudssMatrixCreateDn(X) failed");
  }

  if (!entry.analyzed) {
    s = cudssExecute(entry.handle, CUDSS_PHASE_ANALYSIS, entry.config,
                     entry.data, A, X, B);
    if (s != CUDSS_STATUS_SUCCESS) {
      cleanup();
      return ffi::Error::Internal("cudssExecute(ANALYSIS) failed");
    }
    s = cudssExecute(entry.handle, CUDSS_PHASE_FACTORIZATION, entry.config,
                     entry.data, A, X, B);
    if (s != CUDSS_STATUS_SUCCESS) {
      cleanup();
      return ffi::Error::Internal("cudssExecute(FACTORIZATION) failed");
    }
    entry.analyzed = true;
  } else {
    s = cudssExecute(entry.handle, CUDSS_PHASE_REFACTORIZATION, entry.config,
                     entry.data, A, X, B);
    if (s != CUDSS_STATUS_SUCCESS) {
      cleanup();
      return ffi::Error::Internal("cudssExecute(REFACTORIZATION) failed");
    }
  }

  s = cudssExecute(entry.handle, CUDSS_PHASE_SOLVE, entry.config, entry.data, A,
                   X, B);
  if (s != CUDSS_STATUS_SUCCESS) {
    cleanup();
    return ffi::Error::Internal("cudssExecute(SOLVE) failed");
  }

  cleanup();
  return ffi::Error::Success();
}
