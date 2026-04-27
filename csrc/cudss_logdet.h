// cuDSS-backed log-determinant exposed as a JAX FFI handler.
// TODO: Here there is quite a lot of code that could be cleaned up!

#pragma once

#include <cstdint>
#include <cstdlib>
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
CudssLogdetImpl(cudaStream_t stream, int64_t matrix_type, int64_t factor_token,
                ffi::Buffer<ffi::S32> row_ptr, ffi::Buffer<ffi::S32> col_idx,
                ffi::Buffer<ffi::F64> data, ffi::ResultBuffer<ffi::F64> out) {
  const auto &rp_dims = row_ptr.dimensions();
  if (rp_dims.size() != 1 || rp_dims[0] < 1) {
    return ffi::Error::InvalidArgument("row_ptr must be 1-D with length n+1");
  }
  const int64_t n = rp_dims[0] - 1;
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
  double *d_dummy = nullptr;
  double *d_diag = nullptr;
  // We use cudaFreeAsync on the same stream so frees are sequenced after the
  // reduction kernel; mixing cudaMallocAsync with synchronous cudaFree races
  // with the in-flight kernel and was the source of an illegal-address bug.
  auto cleanup = [&]() {
    if (A)
      cudssMatrixDestroy(A);
    if (B)
      cudssMatrixDestroy(B);
    if (X)
      cudssMatrixDestroy(X);
    if (d_dummy)
      cudaFreeAsync(d_dummy, stream);
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

  // cudssExecute requires non-null X / B even for ANALYSIS+FACTORIZATION.
  const size_t dummy_bytes = static_cast<size_t>(n) * sizeof(double);
  if (cudaError_t e = cudaMallocAsync(reinterpret_cast<void **>(&d_dummy),
                                      dummy_bytes, stream);
      e != cudaSuccess) {
    cleanup();
    return ffi::Error::Internal(std::string("cudaMallocAsync(dummy): ") +
                                cudaGetErrorString(e));
  }
  s = cudssMatrixCreateDn(&B, n, 1, n, d_dummy, CUDA_R_64F,
                          CUDSS_LAYOUT_COL_MAJOR);
  if (s != CUDSS_STATUS_SUCCESS) {
    cleanup();
    return ffi::Error::Internal("cudssMatrixCreateDn(B) failed");
  }
  s = cudssMatrixCreateDn(&X, n, 1, n, d_dummy, CUDA_R_64F,
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

  // cudssDataGet writes into a device buffer for CUDSS_DATA_DIAG. Allocate
  // it on the same stream so the alloc is sequenced with the factorization.
  if (cudaError_t e = cudaMallocAsync(reinterpret_cast<void **>(&d_diag),
                                      dummy_bytes, stream);
      e != cudaSuccess) {
    cleanup();
    return ffi::Error::Internal(std::string("cudaMallocAsync(diag): ") +
                                cudaGetErrorString(e));
  }

  // First call with a null buffer to query the required size, then resize.
  // The size query is host-only and doesn't touch the stream.
  size_t required = 0;
  s = cudssDataGet(entry.handle, entry.data, CUDSS_DATA_DIAG, nullptr, 0,
                   &required);
  if (s != CUDSS_STATUS_SUCCESS) {
    cleanup();
    return ffi::Error::Internal(
        std::string("cudssDataGet(DIAG) size query failed -> status=") +
        std::to_string(static_cast<int>(s)));
  }
  if (required > dummy_bytes) {
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
                   required ? required : dummy_bytes, &written);
  if (s != CUDSS_STATUS_SUCCESS) {
    cleanup();
    return ffi::Error::Internal(
        std::string("cudssDataGet(DIAG) failed -> status=") +
        std::to_string(static_cast<int>(s)));
  }
  if (written < static_cast<size_t>(n) * sizeof(double)) {
    cleanup();
    return ffi::Error::Internal(
        std::string("cudssDataGet(DIAG) wrote ") + std::to_string(written) +
        " bytes (need at least " +
        std::to_string(static_cast<size_t>(n) * sizeof(double)) +
        "). This cuDSS build may not expose the factor diagonal for the "
        "requested matrix type.");
  }
  // No host-side sync: cudssDataGet's copy and the reduction kernel are
  // both queued on `stream`, so the kernel is automatically ordered after
  // the diag copy completes.
  sparsejax_launch_logabs_sum(stream, d_diag, static_cast<int>(n), scale,
                              out->typed_data());
  if (cudaError_t e = cudaPeekAtLastError(); e != cudaSuccess) {
    cleanup();
    return ffi::Error::Internal(std::string("logabs_sum_kernel launch: ") +
                                cudaGetErrorString(e));
  }

  cleanup();
  return ffi::Error::Success();
}
