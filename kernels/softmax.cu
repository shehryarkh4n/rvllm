// Numerically stable softmax kernel.
// Uses shared memory tree reduction -- correct for any block size (1 to 1024).
//
// Launch config:
//   Grid:  (num_rows, 1, 1)
//   Block: (min(vocab_size, 1024), 1, 1)
//   Shared memory: none (uses static shared arrays)
//
// Each block computes softmax for one row.

#include <float.h>

extern "C"
__global__ void softmax_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int vocab_size
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int n = blockDim.x;

    const float* x = input + row * vocab_size;
    float* y = output + row * vocab_size;

    __shared__ float smem[1024];

    // Pass 1: thread-local max across strided elements
    float local_max = -FLT_MAX;
    for (int i = tid; i < vocab_size; i += stride) {
        local_max = fmaxf(local_max, x[i]);
    }
    smem[tid] = local_max;
    __syncthreads();

    // Tree reduction for max (handles any blockDim including non-power-of-2)
    for (int s = n / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < n) {
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        }
        // Handle odd-sized reductions: fold last element into first
        if (s * 2 < n && tid == 0) {
            smem[0] = fmaxf(smem[0], smem[s * 2]);
        }
        __syncthreads();
    }
    float row_max = smem[0];
    __syncthreads();

    // Pass 2: compute exp(x - max) and thread-local sum
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += stride) {
        float e = expf(x[i] - row_max);
        y[i] = e;
        local_sum += e;
    }
    smem[tid] = local_sum;
    __syncthreads();

    // Tree reduction for sum
    for (int s = n / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < n) {
            smem[tid] += smem[tid + s];
        }
        if (s * 2 < n && tid == 0) {
            smem[0] += smem[s * 2];
        }
        __syncthreads();
    }
    float inv_sum = 1.0f / smem[0];

    // Pass 3: normalize
    for (int i = tid; i < vocab_size; i += stride) {
        y[i] *= inv_sum;
    }
}
