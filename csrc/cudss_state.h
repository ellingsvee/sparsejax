// Persistent cuDSS state cache shared by the FFI handlers.

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "cudss.h"
#include <cuda_runtime.h>

namespace sparsejax {

struct CudssEntry {
  cudssHandle_t handle = nullptr;
  cudssConfig_t config = nullptr;
  cudssData_t data = nullptr;
  int64_t n = 0;
  int64_t nnz = 0;
  int matrix_type = -1; // last-used cuDSS matrix type encoding
  bool analyzed = false;

  ~CudssEntry() {
    if (data && handle)
      cudssDataDestroy(handle, data);
    if (config)
      cudssConfigDestroy(config);
    if (handle)
      cudssDestroy(handle);
  }
};

class CudssCache {
public:
  static CudssCache &Instance() {
    static CudssCache c;
    return c;
  }

  // Returns the entry for `token`, creating an empty one if missing. The
  // caller takes the global mutex (returned via `lock`) for the duration of
  // its work with the entry. We keep the simple single-mutex scheme — the
  // FFI handler's hot path (factorize + solve) is GPU-bound so contention
  // is not a real concern at the rates JAX would call us.
  CudssEntry &GetOrCreate(int64_t token, std::unique_lock<std::mutex> &lock) {
    lock = std::unique_lock<std::mutex>(mu_);
    auto it = map_.find(token);
    if (it == map_.end()) {
      it = map_.emplace(token, std::make_unique<CudssEntry>()).first;
    }
    return *it->second;
  }

  // Drop a cached entry (e.g. when the Python-side pattern is GC'd).
  void Drop(int64_t token) {
    std::lock_guard<std::mutex> lk(mu_);
    map_.erase(token);
  }

private:
  std::mutex mu_;
  std::unordered_map<int64_t, std::unique_ptr<CudssEntry>> map_;
};

// Convenience: free-function shim around CudssCache::Drop, exposed to Python.
inline void CudssCacheDrop(int64_t token) {
  CudssCache::Instance().Drop(token);
}

} // namespace sparsejax
