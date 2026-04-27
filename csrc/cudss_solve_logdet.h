// cuDSS-backed combined sparse solve + log-determinant FFI handler.

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include "cudss.h"
#include "cudss_spd_solve.h" // brings in EnsureCudssState + macros
#include "cudss_state.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

extern "C" void sparsejax_launch_logabs_sum(cudaStream_t stream,
                                            const double *d_diag, int n,
                                            double scale, double *d_out);

inline ffi::Error
CudssSolveLogdetImpl(cudaStream_t stream, int64_t matrix_type,
                     int64_t factor_token, ffi::Buffer<ffi::S32> row_ptr,
                     ffi::Buffer<ffi::S32> col_idx, ffi::Buffer<ffi::F64> data,
                     ffi::Buffer<ffi::F64> b, ffi::ResultBuffer<ffi::F64> x,
                     ffi::ResultBuffer<ffi::F64> logdet) {
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
  double scale;
  switch (matrix_type) {
  case 1:
    scale = 2.0;
    break; // SPD: det = prod(L_ii)^2
  case 2:
    scale = 1.0;
    break; // LDL^T: det = prod(D_ii)
  case 0:
  default:
    scale = 1.0;
    break; // LU: prod(U_ii); loses sign
  }

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

  cudssMatrix_t A = nullptr, B = nullptr, X = nullptr;
  double *d_diag = nullptr;
  auto cleanup = [&]() {
    if (A)
      cudssMatrixDestroy(A);
    if (B)
      cudssMatrixDestroy(B);
    if (X)
      cudssMatrixDestroy(X);
    if (d_diag)
      cudaFreeAsync(d_diag, stream);
  };

  cudssStatus_t s = cudssMatrixCreateCsr(
      &A, n, n, nnz, row_ptr.typed_data(), nullptr, col_idx.typed_data(),
      data.typed_data(), CUDA_R_32I, CUDA_R_64F, mtype, mview, CUDSS_BASE_ZERO);
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

  const size_t diag_bytes = static_cast<size_t>(n) * sizeof(double);
  if (cudaError_t e = cudaMallocAsync(reinterpret_cast<void **>(&d_diag),
                                      diag_bytes, stream);
      e != cudaSuccess) {
    cleanup();
    return ffi::Error::Internal(std::string("cudaMallocAsync(diag): ") +
                                cudaGetErrorString(e));
  }

  size_t required = 0;
  s = cudssDataGet(entry.handle, entry.data, CUDSS_DATA_DIAG, nullptr, 0,
                   &required);
  if (s != CUDSS_STATUS_SUCCESS) {
    cleanup();
    return ffi::Error::Internal(
        std::string("cudssDataGet(DIAG) size query failed -> status=") +
        std::to_string(static_cast<int>(s)));
  }
  if (required > diag_bytes) {
    cudaFreeAsync(d_diag, stream);
    d_diag = nullptr;
    if (cudaError_t e = cudaMallocAsync(reinterpret_cast<void **>(&d_diag),
                                        required, stream);
        e != cudaSuccess) {
      cleanup();
      return ffi::Error::Internal(std::string("cudaMallocAsync(diag,grow): ") +
                                  cudaGetErrorString(e));
    }
  }
  size_t written = 0;
  s = cudssDataGet(entry.handle, entry.data, CUDSS_DATA_DIAG, d_diag,
                   required ? required : diag_bytes, &written);
  if (s != CUDSS_STATUS_SUCCESS) {
    cleanup();
    return ffi::Error::Internal(
        std::string("cudssDataGet(DIAG) failed -> status=") +
        std::to_string(static_cast<int>(s)));
  }
  if (written < diag_bytes) {
    cleanup();
    return ffi::Error::Internal(
        std::string("cudssDataGet(DIAG) wrote ") + std::to_string(written) +
        " bytes (need at least " + std::to_string(diag_bytes) +
        "). This cuDSS build may not expose the factor diagonal for the "
        "requested matrix type.");
  }

  sparsejax_launch_logabs_sum(stream, d_diag, static_cast<int>(n), scale,
                              logdet->typed_data());
  if (cudaError_t e = cudaPeekAtLastError(); e != cudaSuccess) {
    cleanup();
    return ffi::Error::Internal(std::string("logabs_sum_kernel launch: ") +
                                cudaGetErrorString(e));
  }

  cleanup();
  return ffi::Error::Success();
}
