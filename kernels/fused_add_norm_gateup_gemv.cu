// Fused residual-add + RMSNorm + GEMV kernel for gate_up projection (M=1 decode).
//
// Eliminates intermediate buffers by computing:
//   residual_out = input + add_vec
//   normed = rmsnorm(residual_out, norm_weight, eps)
//   output[j] = dot(proj_weight[j, :], normed)
//
// Each block handles rpb=8 output rows. All blocks redundantly compute the
// residual add + RMSNorm into shared memory, then each block computes 8 dot
// products against consecutive weight rows.
//
// Launch config:
//   Grid:  ((gate_up_dim + 7) / 8, 1, 1)
//   Block: (256, 1, 1)
//   Shared mem: hidden_size * sizeof(float) + 8 * sizeof(float)
//
// For Qwen2.5-1.5B: hidden=1536, gate_up_dim=17920, 2240 blocks.

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define THREADS 256
#define ROWS_PER_BLOCK 8

__device__ __forceinline__ float warp_reduce_sum_angv(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

extern "C"
__global__ void __launch_bounds__(THREADS)
fused_cute_add_norm_gateup_gemv(
    __half* __restrict__ output,           // [gate_up_dim]
    __half* __restrict__ residual_out,     // [hidden_size]
    const __half* __restrict__ input,      // [hidden_size] -- residual
    const __half* __restrict__ add_vec,    // [hidden_size] -- O projection output
    const __half* __restrict__ norm_weight, // [hidden_size]
    const __half* __restrict__ proj_weight, // [gate_up_dim, hidden_size] row-major
    float eps,
    int hidden_size,
    int gate_up_dim
) {
    const int block_row_base = blockIdx.x * ROWS_PER_BLOCK;
    if (block_row_base >= gate_up_dim) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int NUM_WARPS = THREADS / 32;

    // Shared memory layout:
    //   [0 .. hidden_size-1]  : normed hidden state (f32)
    //   [hidden_size .. hidden_size+7] : warp partial sums scratch
    extern __shared__ float smem[];
    float* s_normed = smem;
    float* s_warp   = smem + hidden_size;

    // ---- Phase 1: Residual add + RMSNorm ----
    // All blocks redundantly compute this. Cheap for hidden_size=1536.

    const int h2 = hidden_size / 2;
    const half2* in2  = (const half2*)input;
    const half2* add2 = (const half2*)add_vec;

    float local_ss = 0.0f;
    for (int i = tid; i < h2; i += THREADS) {
        half2 a = in2[i];
        half2 b = add2[i];
        float v0 = __half2float(a.x) + __half2float(b.x);
        float v1 = __half2float(a.y) + __half2float(b.y);
        s_normed[i * 2]     = v0;
        s_normed[i * 2 + 1] = v1;
        local_ss += v0 * v0 + v1 * v1;
    }
    // Handle odd hidden_size
    if ((hidden_size & 1) && tid == 0) {
        int last = hidden_size - 1;
        float v = __half2float(input[last]) + __half2float(add_vec[last]);
        s_normed[last] = v;
        local_ss += v * v;
    }

    // Warp-level reduction of sum-of-squares
    local_ss = warp_reduce_sum_angv(local_ss);
    if (lane_id == 0) s_warp[warp_id] = local_ss;
    __syncthreads();

    // Cross-warp reduction in first warp
    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_warp[lane_id] : 0.0f;
        val = warp_reduce_sum_angv(val);
        if (lane_id == 0) {
            s_warp[0] = rsqrtf(val / (float)hidden_size + eps);
        }
    }
    __syncthreads();

    float rms_scale = s_warp[0];

    // Block 0 writes residual_out (only one block to avoid races)
    if (blockIdx.x == 0) {
        for (int i = tid; i < hidden_size; i += THREADS) {
            residual_out[i] = __float2half(s_normed[i]);
        }
    }

    // Apply norm weights in-place in smem
    for (int i = tid; i < hidden_size; i += THREADS) {
        s_normed[i] = s_normed[i] * __half2float(norm_weight[i]) * rms_scale;
    }
    __syncthreads();

    // ---- Phase 2: GEMV -- warp-per-row, 128-bit vectorized weight loads ----
    {
        const int row = block_row_base + warp_id;
        if (row < gate_up_dim) {
            const int4* w4 = (const int4*)(proj_weight + (long long)row * hidden_size);
            const int h8 = hidden_size / 8;
            float acc = 0.0f;
            #pragma unroll 4
            for (int i = lane_id; i < h8; i += 32) {
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
            for (int i = h8 * 8 + lane_id; i < hidden_size; i += 32)
                acc += __half2float(proj_weight[(long long)row * hidden_size + i]) * s_normed[i];
            acc = warp_reduce_sum_angv(acc);
            if (lane_id == 0) {
                output[row] = __float2half(acc);
            }
        }
    }
}

// --------------------------------------------------------------------------
// FP8 variant: weights stored as E4M3 bytes with per-row scale
// --------------------------------------------------------------------------
extern "C"
__global__ void __launch_bounds__(THREADS)
fused_cute_add_norm_gateup_fp8_gemv(
    __half* __restrict__ output,
    __half* __restrict__ residual_out,
    const __half* __restrict__ input,
    const __half* __restrict__ add_vec,
    const __half* __restrict__ norm_weight,
    const unsigned char* __restrict__ proj_weight,
    const __half* __restrict__ weight_scale,
    float eps,
    int hidden_size,
    int gate_up_dim
) {
    const int block_row_base = blockIdx.x * ROWS_PER_BLOCK;
    if (block_row_base >= gate_up_dim) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int NUM_WARPS = THREADS / 32;

    extern __shared__ float smem[];
    float* s_normed = smem;
    float* s_warp   = smem + hidden_size;

    // ---- Phase 1: Residual add + RMSNorm (identical to f16) ----
    const int h2 = hidden_size / 2;
    const half2* in2  = (const half2*)input;
    const half2* add2 = (const half2*)add_vec;

    float local_ss = 0.0f;
    for (int i = tid; i < h2; i += THREADS) {
        half2 a = in2[i];
        half2 b = add2[i];
        float v0 = __half2float(a.x) + __half2float(b.x);
        float v1 = __half2float(a.y) + __half2float(b.y);
        s_normed[i * 2]     = v0;
        s_normed[i * 2 + 1] = v1;
        local_ss += v0 * v0 + v1 * v1;
    }
    if ((hidden_size & 1) && tid == 0) {
        int last = hidden_size - 1;
        float v = __half2float(input[last]) + __half2float(add_vec[last]);
        s_normed[last] = v;
        local_ss += v * v;
    }

    local_ss = warp_reduce_sum_angv(local_ss);
    if (lane_id == 0) s_warp[warp_id] = local_ss;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_warp[lane_id] : 0.0f;
        val = warp_reduce_sum_angv(val);
        if (lane_id == 0) {
            s_warp[0] = rsqrtf(val / (float)hidden_size + eps);
        }
    }
    __syncthreads();

    float rms_scale = s_warp[0];

    if (blockIdx.x == 0) {
        for (int i = tid; i < hidden_size; i += THREADS) {
            residual_out[i] = __float2half(s_normed[i]);
        }
    }

    for (int i = tid; i < hidden_size; i += THREADS) {
        s_normed[i] = s_normed[i] * __half2float(norm_weight[i]) * rms_scale;
    }
    __syncthreads();

    // ---- Phase 2: GEMV with FP8 weights -- warp-per-row ----
    {
        const int row = block_row_base + warp_id;
        if (row < gate_up_dim) {
            const unsigned char* w_row = proj_weight + (long long)row * hidden_size;
            float row_sc = __half2float(weight_scale[row]);
            float acc = 0.0f;
            const int h4 = hidden_size / 4;
            for (int i = lane_id; i < h4; i += 32) {
                int base = i * 4;
                unsigned int packed = *reinterpret_cast<const unsigned int*>(w_row + base);
                unsigned char b0 = packed & 0xFF;
                unsigned char b1 = (packed >> 8) & 0xFF;
                unsigned char b2 = (packed >> 16) & 0xFF;
                unsigned char b3 = (packed >> 24) & 0xFF;
                float w0 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b0)) * row_sc;
                float w1 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b1)) * row_sc;
                float w2 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b2)) * row_sc;
                float w3 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b3)) * row_sc;
                acc += w0 * s_normed[base]
                     + w1 * s_normed[base + 1]
                     + w2 * s_normed[base + 2]
                     + w3 * s_normed[base + 3];
            }
            // Handle remaining elements
            for (int i = h4 * 4 + lane_id; i < hidden_size; i += 32) {
                unsigned char b = w_row[i];
                float w = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b)) * row_sc;
                acc += w * s_normed[i];
            }
            acc = warp_reduce_sum_angv(acc);
            if (lane_id == 0) {
                output[row] = __float2half(acc);
            }
        }
    }
}
