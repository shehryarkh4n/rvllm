// Custom f16 GEMV kernel for M=1 decode.
//
// y[n] = A[n, k] @ x[k]  (f16 input/output, f32 accumulation)
//
// At M=1, cuBLAS GEMM wastes 99% of its tile capacity (128x128 tiles
// with only 1 row filled). This kernel is specialized for vectors:
// - One thread block per output element (or small group)
// - Vectorized f16 loads (half2) for 2x bandwidth
// - Warp-level reduction for the dot product
// - f32 accumulation for numerical stability
//
// Launch config:
//   Grid:  (n, 1, 1)  -- one block per output element
//   Block: (256, 1, 1) -- 256 threads reduce over k dimension
//   Shared memory: 0 (warp shuffle reduction)
//
// For Qwen2.5-1.5B shapes:
//   QKV fused:  n=2048,  k=1536
//   O proj:     n=1536,  k=1536
//   gate+up:    n=17920, k=1536
//   down:       n=1536,  k=8960

#include <cuda_fp16.h>

#define GEMV_THREADS 256

__device__ __forceinline__ float warp_reduce_sum_gemv(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// y[n] = A[n, k] @ x[k], A is row-major [n, k]
extern "C"
__global__ void gemv_f16_kernel(
    __half* __restrict__ output,      // [n]
    const __half* __restrict__ weight, // [n, k] row-major
    const __half* __restrict__ input,  // [k]
    int n,
    int k
) {
    const int row = blockIdx.x;
    if (row >= n) return;

    const int tid = threadIdx.x;
    const __half* row_ptr = weight + row * k;

    // Each thread accumulates a partial dot product over k/THREADS elements
    float acc = 0.0f;

    // Vectorized loads: process 2 elements at a time via half2
    const int k2 = k / 2;
    const half2* row2 = (const half2*)row_ptr;
    const half2* in2  = (const half2*)input;

    for (int i = tid; i < k2; i += GEMV_THREADS) {
        half2 w = row2[i];
        half2 x = in2[i];
        // f32 accumulation: w.x*x.x + w.y*x.y
        acc += __half2float(w.x) * __half2float(x.x);
        acc += __half2float(w.y) * __half2float(x.y);
    }

    // Handle odd k
    if (k % 2 != 0) {
        int idx = k2 * 2 + tid;
        if (idx < k) {
            acc += __half2float(row_ptr[idx]) * __half2float(input[idx]);
        }
    }

    // Warp-level reduction
    acc = warp_reduce_sum_gemv(acc);

    // Cross-warp reduction via shared memory
    __shared__ float warp_sums[GEMV_THREADS / 32];
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    if (lane_id == 0) {
        warp_sums[warp_id] = acc;
    }
    __syncthreads();

    // Final reduction in first warp
    if (warp_id == 0) {
        float val = (lane_id < (GEMV_THREADS / 32)) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum_gemv(val);
        if (lane_id == 0) {
            output[row] = __float2half(val);
        }
    }
}

// Batched variant: Y[m, n] = X[m, k] @ A[n, k]^T
// For small M (2-32), each block handles one (m, n_chunk) pair.
// This is more efficient than launching M separate GEMV calls.
extern "C"
__global__ void gemv_batched_f16_kernel(
    __half* __restrict__ output,       // [m, n]
    const __half* __restrict__ weight,  // [n, k] row-major (shared across batch)
    const __half* __restrict__ input,   // [m, k] row-major
    int m,
    int n,
    int k
) {
    const int row = blockIdx.x;  // output row in n dimension
    const int batch = blockIdx.y; // batch index in m dimension
    if (row >= n || batch >= m) return;

    const int tid = threadIdx.x;
    const __half* w_row = weight + row * k;
    const __half* x_row = input + batch * k;

    float acc = 0.0f;
    const int k2 = k / 2;
    const half2* w2 = (const half2*)w_row;
    const half2* x2 = (const half2*)x_row;

    for (int i = tid; i < k2; i += GEMV_THREADS) {
        half2 w = w2[i];
        half2 x = x2[i];
        acc += __half2float(w.x) * __half2float(x.x);
        acc += __half2float(w.y) * __half2float(x.y);
    }

    if (k % 2 != 0) {
        int idx = k2 * 2 + tid;
        if (idx < k) {
            acc += __half2float(w_row[idx]) * __half2float(x_row[idx]);
        }
    }

    acc = warp_reduce_sum_gemv(acc);

    __shared__ float warp_sums[GEMV_THREADS / 32];
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    if (lane_id == 0) warp_sums[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < (GEMV_THREADS / 32)) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum_gemv(val);
        if (lane_id == 0) {
            output[batch * n + row] = __float2half(val);
        }
    }
}
