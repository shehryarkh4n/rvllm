// Fused RMSNorm + GEMV kernel for M=1 decode (f16 I/O, f32 accumulation).
//
// Eliminates the intermediate normed-hidden-state buffer and one kernel launch
// by computing RMSNorm inline before the dot product.
//
//   output[j] = sum_i( rmsnorm(hidden)[i] * proj_weight[j, i] )
//
// where rmsnorm(x)[i] = (x[i] / rms) * norm_weight[i]
//       rms = sqrt(mean(x^2) + eps)
//
// Strategy:
//   1. All threads cooperatively load hidden[hidden_size] into shared memory
//      and compute sum-of-squares for RMSNorm (parallel reduction).
//   2. Each block handles one output row j. Threads compute the dot product
//      of the normalized hidden vector with proj_weight[j, :].
//   3. Warp-shuffle + shared-memory reduction to produce the final scalar.
//
// Launch config:
//   Grid:  (out_dim, 1, 1)    -- one block per output element
//   Block: (THREADS, 1, 1)    -- THREADS=256
//   Shared memory: hidden_size * sizeof(float) + (THREADS/32) * sizeof(float)
//
// For Qwen2.5-1.5B:
//   QKV fused:  out_dim=2048,  hidden_size=1536
//   O proj:     out_dim=1536,  hidden_size=1536
//   gate+up:    out_dim=17920, hidden_size=1536

#include <cuda_fp16.h>

#define FUSED_NORM_GEMV_THREADS 256

__device__ __forceinline__ float warp_reduce_sum_fng(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

extern "C"
__global__ void __launch_bounds__(FUSED_NORM_GEMV_THREADS)
fused_norm_gemv_f16_kernel(
    __half* __restrict__ output,           // [out_dim]
    const __half* __restrict__ hidden,     // [hidden_size]
    const __half* __restrict__ norm_weight, // [hidden_size]
    const __half* __restrict__ proj_weight, // [out_dim, hidden_size] row-major
    float eps,
    int out_dim,
    int hidden_size
) {
    const int row = blockIdx.x;
    if (row >= out_dim) return;

    const int tid = threadIdx.x;

    // Shared memory layout:
    //   [0 .. hidden_size-1]       : normalized hidden state (f32)
    //   [hidden_size .. hidden_size + WARPS-1] : warp partial sums for dot product
    extern __shared__ float smem[];
    float* s_normed = smem;                                        // [hidden_size]
    float* s_warp   = smem + hidden_size;                          // [FUSED_NORM_GEMV_THREADS / 32]

    // --- Phase 1: RMSNorm ---
    // All blocks redundantly compute RMSNorm of hidden into shared memory.
    // This is cheap (hidden_size=1536, one pass to load + reduce, one pass to normalize)
    // and avoids writing an intermediate global buffer.

    // 1a. Load hidden into shared mem and compute local sum-of-squares
    float local_ss = 0.0f;
    for (int i = tid; i < hidden_size; i += FUSED_NORM_GEMV_THREADS) {
        float val = __half2float(hidden[i]);
        s_normed[i] = val;  // temporarily store raw values
        local_ss += val * val;
    }

    // 1b. Warp-level reduction of sum-of-squares
    local_ss = warp_reduce_sum_fng(local_ss);

    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int NUM_WARPS = FUSED_NORM_GEMV_THREADS / 32;

    if (lane_id == 0) {
        s_warp[warp_id] = local_ss;
    }
    __syncthreads();

    // Final reduction in first warp
    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_warp[lane_id] : 0.0f;
        val = warp_reduce_sum_fng(val);
        if (lane_id == 0) {
            // Store rms_scale in s_warp[0] for all threads to read
            s_warp[0] = rsqrtf(val / (float)hidden_size + eps);
        }
    }
    __syncthreads();

    float rms_scale = s_warp[0];

    // 1c. Apply normalization: s_normed[i] = hidden[i] * norm_weight[i] * rms_scale
    for (int i = tid; i < hidden_size; i += FUSED_NORM_GEMV_THREADS) {
        s_normed[i] = s_normed[i] * __half2float(norm_weight[i]) * rms_scale;
    }
    __syncthreads();

    // --- Phase 2: GEMV dot product ---
    // output[row] = dot(proj_weight[row, :], s_normed[:])
    const int4* w4 = (const int4*)(proj_weight + (long long)row * hidden_size);
    const int h8 = hidden_size / 8;

    float acc = 0.0f;

    // Vectorized int4 (128-bit) loads: 8 halfs per iteration
    #pragma unroll 4
    for (int i = tid; i < h8; i += FUSED_NORM_GEMV_THREADS) {
        int4 packed = w4[i];
        half2 w01 = *reinterpret_cast<half2*>(&packed.x);
        half2 w23 = *reinterpret_cast<half2*>(&packed.y);
        half2 w45 = *reinterpret_cast<half2*>(&packed.z);
        half2 w67 = *reinterpret_cast<half2*>(&packed.w);
        int base = i * 8;
        acc += __half2float(w01.x) * s_normed[base]
             + __half2float(w01.y) * s_normed[base + 1]
             + __half2float(w23.x) * s_normed[base + 2]
             + __half2float(w23.y) * s_normed[base + 3]
             + __half2float(w45.x) * s_normed[base + 4]
             + __half2float(w45.y) * s_normed[base + 5]
             + __half2float(w67.x) * s_normed[base + 6]
             + __half2float(w67.y) * s_normed[base + 7];
    }

    // Handle remainder elements not covered by int4 loads
    for (int i = h8 * 8 + tid; i < hidden_size; i += FUSED_NORM_GEMV_THREADS)
        acc += __half2float(proj_weight[(long long)row * hidden_size + i]) * s_normed[i];

    // Warp-level reduction
    acc = warp_reduce_sum_fng(acc);

    if (lane_id == 0) {
        s_warp[warp_id] = acc;
    }
    __syncthreads();

    // Final reduction in first warp -> write output
    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_warp[lane_id] : 0.0f;
        val = warp_reduce_sum_fng(val);
        if (lane_id == 0) {
            output[row] = __float2half(val);
        }
    }
}

// Variant with bias addition: output[j] = dot(norm(hidden), weight[j,:]) + bias[j]
extern "C"
__global__ void __launch_bounds__(FUSED_NORM_GEMV_THREADS)
fused_norm_gemv_bias_f16_kernel(
    __half* __restrict__ output,           // [out_dim]
    const __half* __restrict__ hidden,     // [hidden_size]
    const __half* __restrict__ norm_weight, // [hidden_size]
    const __half* __restrict__ proj_weight, // [out_dim, hidden_size] row-major
    const __half* __restrict__ bias,        // [out_dim]
    float eps,
    int out_dim,
    int hidden_size
) {
    const int row = blockIdx.x;
    if (row >= out_dim) return;

    const int tid = threadIdx.x;

    extern __shared__ float smem[];
    float* s_normed = smem;
    float* s_warp   = smem + hidden_size;

    // Phase 1: RMSNorm (identical to above)
    float local_ss = 0.0f;
    for (int i = tid; i < hidden_size; i += FUSED_NORM_GEMV_THREADS) {
        float val = __half2float(hidden[i]);
        s_normed[i] = val;
        local_ss += val * val;
    }

    local_ss = warp_reduce_sum_fng(local_ss);

    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int NUM_WARPS = FUSED_NORM_GEMV_THREADS / 32;

    if (lane_id == 0) {
        s_warp[warp_id] = local_ss;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_warp[lane_id] : 0.0f;
        val = warp_reduce_sum_fng(val);
        if (lane_id == 0) {
            s_warp[0] = rsqrtf(val / (float)hidden_size + eps);
        }
    }
    __syncthreads();

    float rms_scale = s_warp[0];

    for (int i = tid; i < hidden_size; i += FUSED_NORM_GEMV_THREADS) {
        s_normed[i] = s_normed[i] * __half2float(norm_weight[i]) * rms_scale;
    }
    __syncthreads();

    // Phase 2: GEMV dot product + bias
    const int4* w4 = (const int4*)(proj_weight + (long long)row * hidden_size);
    const int h8 = hidden_size / 8;

    float acc = 0.0f;

    // Vectorized int4 (128-bit) loads: 8 halfs per iteration
    #pragma unroll 4
    for (int i = tid; i < h8; i += FUSED_NORM_GEMV_THREADS) {
        int4 packed = w4[i];
        half2 w01 = *reinterpret_cast<half2*>(&packed.x);
        half2 w23 = *reinterpret_cast<half2*>(&packed.y);
        half2 w45 = *reinterpret_cast<half2*>(&packed.z);
        half2 w67 = *reinterpret_cast<half2*>(&packed.w);
        int base = i * 8;
        acc += __half2float(w01.x) * s_normed[base]
             + __half2float(w01.y) * s_normed[base + 1]
             + __half2float(w23.x) * s_normed[base + 2]
             + __half2float(w23.y) * s_normed[base + 3]
             + __half2float(w45.x) * s_normed[base + 4]
             + __half2float(w45.y) * s_normed[base + 5]
             + __half2float(w67.x) * s_normed[base + 6]
             + __half2float(w67.y) * s_normed[base + 7];
    }

    // Handle remainder elements not covered by int4 loads
    for (int i = h8 * 8 + tid; i < hidden_size; i += FUSED_NORM_GEMV_THREADS)
        acc += __half2float(proj_weight[(long long)row * hidden_size + i]) * s_normed[i];

    acc = warp_reduce_sum_fng(acc);

    if (lane_id == 0) {
        s_warp[warp_id] = acc;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_warp[lane_id] : 0.0f;
        val = warp_reduce_sum_fng(val);
        if (lane_id == 0) {
            output[row] = __float2half(val + __half2float(bias[row]));
        }
    }
}
