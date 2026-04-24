// GPU reduction: out = scale * sum_i log(|diag[i]|).
//
// Used by the cuDSS logdet FFI handler to turn the factor diagonal queried
// via cudssDataGet(CUDSS_DATA_DIAG) into a log-determinant on-device,
// without a host round-trip.

#include <cuda_runtime.h>
#include <math.h>

namespace {

constexpr int kBlock = 256;

__global__ void logabs_sum_kernel(const double *__restrict__ diag, int n,
                                  double scale, double *out) {
  __shared__ double smem[kBlock];
  const int tid = threadIdx.x;
  double acc = 0.0;
  for (int i = tid; i < n; i += blockDim.x) {
    double v = diag[i];
    if (v != 0.0) {
      acc += log(fabs(v));
    }
  }
  smem[tid] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      smem[tid] += smem[tid + s];
    __syncthreads();
  }
  if (tid == 0)
    *out = smem[0] * scale;
}

} // namespace

extern "C" void sparsejax_launch_logabs_sum(cudaStream_t stream,
                                            const double *d_diag, int n,
                                            double scale, double *d_out) {
  logabs_sum_kernel<<<1, kBlock, 0, stream>>>(d_diag, n, scale, d_out);
}
