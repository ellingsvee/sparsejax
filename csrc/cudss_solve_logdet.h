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
extern "C" void sparsejax_launch_logabs_sum_f32(cudaStream_t stream,
                                                const float *d_diag, int n,
                                                double scale, float *d_out);

template <typename T, ffi::DataType FfiT>
inline ffi::Error
CudssSolveLogdetImpl(cudaStream_t stream, int64_t matrix_type,
                     int64_t factor_token, int64_t reordering_alg,
                     int64_t factorization_alg, int64_t solve_alg,
                     int64_t use_matching, int64_t host_nthreads,
                     int64_t use_superpanels, ffi::Buffer<ffi::S32> row_ptr,
                     ffi::Buffer<ffi::S32> col_idx, ffi::Buffer<FfiT> data,
                     ffi::Buffer<FfiT> b, ffi::ResultBuffer<FfiT> x,
                     ffi::ResultBuffer<FfiT> logdet) {
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

  cudssMatrix_t A = nullptr, B = nullptr, X = nullptr;
  T *d_diag = nullptr;
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
      data.typed_data(), CUDA_R_32I, sparsejax::CudaValueType<T>(), mtype, mview,
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

  const size_t diag_bytes = static_cast<size_t>(n) * sizeof(T);
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
        std::string("cudssDataGet(DIAG) size query failed: ") +
        sparsejax::CudssStatusString(s) + " (" +
        std::to_string(static_cast<int>(s)) + ")");
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
        std::string("cudssDataGet(DIAG) failed: ") +
        sparsejax::CudssStatusString(s) + " (" +
        std::to_string(static_cast<int>(s)) + ")");
  }
  if (written < diag_bytes) {
    cleanup();
    return ffi::Error::Internal(
        std::string("cudssDataGet(DIAG) wrote ") + std::to_string(written) +
        " bytes (need at least " + std::to_string(diag_bytes) +
        "). This cuDSS build may not expose the factor diagonal for the "
        "requested matrix type.");
  }

  if constexpr (std::is_same_v<T, float>) {
    sparsejax_launch_logabs_sum_f32(stream, d_diag, static_cast<int>(n), scale,
                                    logdet->typed_data());
  } else {
    sparsejax_launch_logabs_sum(stream, d_diag, static_cast<int>(n), scale,
                                logdet->typed_data());
  }
  if (cudaError_t e = cudaPeekAtLastError(); e != cudaSuccess) {
    cleanup();
    return ffi::Error::Internal(std::string("logabs_sum_kernel launch: ") +
                                cudaGetErrorString(e));
  }

  cleanup();
  return ffi::Error::Success();
}
