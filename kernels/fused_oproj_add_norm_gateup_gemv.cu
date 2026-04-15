// Mega-fused O-projection + residual-add + RMSNorm + gate_up GEMV kernel (M=1 decode).
//
// Eliminates the intermediate attn_proj HBM write+read by computing:
//   oproj[h] = dot(o_weight[h, :], attn_out)      -- O projection
//   residual_out = residual + oproj                 -- residual add
//   normed = rmsnorm(residual_out, norm_weight, eps) -- RMSNorm
//   gate_up_out[j] = dot(gateup_weight[j, :], normed) -- gate_up GEMV
//
// Phase 1: All blocks redundantly compute O-proj into shared memory.
//   O_weight fits in L2 (~4.7MB for hidden=1536), so redundant reads are cheap.
// Phase 2: Residual add + RMSNorm in shared memory.
// Phase 3: Each block computes RPB=8 rows of gate_up GEMV.
//
// Launch config:
//   Grid:  ((gate_up_dim + RPB - 1) / RPB, 1, 1)
//   Block: (256, 1, 1)
//   Shared mem: hidden_size * sizeof(float) + 8 * sizeof(float)

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define THREADS 256
#define RPB 8

__device__ __forceinline__ float warp_reduce_sum_opag(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

extern "C"
__global__ void __launch_bounds__(THREADS)
fused_cute_oproj_add_norm_gateup_gemv(
    __half* __restrict__ gate_up_out,       // [gate_up_dim]
    __half* __restrict__ residual_out,      // [hidden_size]
    const __half* __restrict__ attn_out,    // [q_dim]
    const __half* __restrict__ o_weight,    // [hidden_size, q_dim] row-major
    const __half* __restrict__ residual,    // [hidden_size]
    const __half* __restrict__ norm_weight, // [hidden_size]
    const __half* __restrict__ gateup_weight, // [gate_up_dim, hidden_size] row-major
    float eps,
    int q_dim,
    int hidden_size,
    int gate_up_dim
) {
    const int block_row_base = blockIdx.x * RPB;
    if (block_row_base >= gate_up_dim) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int NUM_WARPS = THREADS / 32;

    // Shared memory layout:
    //   [0 .. hidden_size-1]      : O-proj result / normed hidden state (f32)
    //   [hidden_size .. hidden_size+7] : warp partial sums scratch
    extern __shared__ float smem[];
    float* s_hidden = smem;
    float* s_warp   = smem + hidden_size;

    // ---- Phase 1: O-projection GEMV into shared memory ----
    // Each block computes all hidden_size output elements.
    // For each output element h: oproj[h] = dot(o_weight[h,:], attn_out)
    // attn_out is small (q_dim * 2 bytes), will be cached in L1 after first pass.
    // o_weight rows are streamed; the full matrix fits in L2 for small models.

    const int q8 = q_dim / 8;
    const int4* attn4 = (const int4*)attn_out;

    for (int h = 0; h < hidden_size; h++) {
        const int4* ow4 = (const int4*)(o_weight + (long long)h * q_dim);
        float acc = 0.0f;

        #pragma unroll 4
        for (int i = tid; i < q8; i += THREADS) {
            int4 packed_w = ow4[i];
            half2 w01 = *reinterpret_cast<half2*>(&packed_w.x);
            half2 w23 = *reinterpret_cast<half2*>(&packed_w.y);
            half2 w45 = *reinterpret_cast<half2*>(&packed_w.z);
            half2 w67 = *reinterpret_cast<half2*>(&packed_w.w);

            int4 packed_a = attn4[i];
            half2 a01 = *reinterpret_cast<half2*>(&packed_a.x);
            half2 a23 = *reinterpret_cast<half2*>(&packed_a.y);
            half2 a45 = *reinterpret_cast<half2*>(&packed_a.z);
            half2 a67 = *reinterpret_cast<half2*>(&packed_a.w);

            acc += __half2float(w01.x) * __half2float(a01.x) + __half2float(w01.y) * __half2float(a01.y)
                 + __half2float(w23.x) * __half2float(a23.x) + __half2float(w23.y) * __half2float(a23.y)
                 + __half2float(w45.x) * __half2float(a45.x) + __half2float(w45.y) * __half2float(a45.y)
                 + __half2float(w67.x) * __half2float(a67.x) + __half2float(w67.y) * __half2float(a67.y);
        }
        for (int i = q8 * 8 + tid; i < q_dim; i += THREADS)
            acc += __half2float(o_weight[(long long)h * q_dim + i]) * __half2float(attn_out[i]);

        // Warp reduction
        acc = warp_reduce_sum_opag(acc);
        if (lane_id == 0) s_warp[warp_id] = acc;
        __syncthreads();

        // Cross-warp reduction in first warp
        if (warp_id == 0) {
            float val = (lane_id < NUM_WARPS) ? s_warp[lane_id] : 0.0f;
            val = warp_reduce_sum_opag(val);
            if (lane_id == 0) {
                s_hidden[h] = val;
            }
        }
        // Must sync before next iteration reuses s_warp
        if (h + 1 < hidden_size) __syncthreads();
    }
    __syncthreads();

    // ---- Phase 2: Residual add + RMSNorm ----
    const int h2 = hidden_size / 2;
    const half2* res2 = (const half2*)residual;

    float local_ss = 0.0f;
    for (int i = tid; i < h2; i += THREADS) {
        half2 r = res2[i];
        float v0 = s_hidden[i * 2]     + __half2float(r.x);
        float v1 = s_hidden[i * 2 + 1] + __half2float(r.y);
        s_hidden[i * 2]     = v0;
        s_hidden[i * 2 + 1] = v1;
        local_ss += v0 * v0 + v1 * v1;
    }
    if ((hidden_size & 1) && tid == 0) {
        int last = hidden_size - 1;
        float v = s_hidden[last] + __half2float(residual[last]);
        s_hidden[last] = v;
        local_ss += v * v;
    }

    // Warp reduction of sum-of-squares
    local_ss = warp_reduce_sum_opag(local_ss);
    if (lane_id == 0) s_warp[warp_id] = local_ss;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_warp[lane_id] : 0.0f;
        val = warp_reduce_sum_opag(val);
        if (lane_id == 0) {
            s_warp[0] = rsqrtf(val / (float)hidden_size + eps);
        }
    }
    __syncthreads();

    float rms_scale = s_warp[0];

    // Block 0 writes residual_out (pre-norm residual for next layer)
    if (blockIdx.x == 0) {
        for (int i = tid; i < hidden_size; i += THREADS) {
            residual_out[i] = __float2half(s_hidden[i]);
        }
    }

    // Apply norm weights in-place in smem
    for (int i = tid; i < hidden_size; i += THREADS) {
        s_hidden[i] = s_hidden[i] * __half2float(norm_weight[i]) * rms_scale;
    }
    __syncthreads();

    // ---- Phase 3: gate_up GEMV -- RPB=8 dot products per block ----
    const int rows_this_block = min(RPB, gate_up_dim - block_row_base);
    const int h8 = hidden_size / 8;

    float acc[RPB];
    #pragma unroll
    for (int r = 0; r < RPB; r++) acc[r] = 0.0f;

    #pragma unroll 4
    for (int i = tid; i < h8; i += THREADS) {
        int base = i * 8;
        float s0 = s_hidden[base],     s1 = s_hidden[base+1],
              s2 = s_hidden[base+2],   s3 = s_hidden[base+3],
              s4 = s_hidden[base+4],   s5 = s_hidden[base+5],
              s6 = s_hidden[base+6],   s7 = s_hidden[base+7];
        #pragma unroll
        for (int r = 0; r < RPB; r++) {
            if (r < rows_this_block) {
                const int4* w4 = (const int4*)(gateup_weight + (long long)(block_row_base + r) * hidden_size);
                int4 packed = w4[i];
                half2 w01 = *reinterpret_cast<half2*>(&packed.x);
                half2 w23 = *reinterpret_cast<half2*>(&packed.y);
                half2 w45 = *reinterpret_cast<half2*>(&packed.z);
                half2 w67 = *reinterpret_cast<half2*>(&packed.w);
                acc[r] += __half2float(w01.x) * s0 + __half2float(w01.y) * s1
                        + __half2float(w23.x) * s2 + __half2float(w23.y) * s3
                        + __half2float(w45.x) * s4 + __half2float(w45.y) * s5
                        + __half2float(w67.x) * s6 + __half2float(w67.y) * s7;
            }
        }
    }

    // Handle remainder elements
    for (int i = h8 * 8 + tid; i < hidden_size; i += THREADS) {
        float sn = s_hidden[i];
        #pragma unroll
        for (int r = 0; r < RPB; r++) {
            if (r < rows_this_block) {
                acc[r] += __half2float(gateup_weight[(long long)(block_row_base + r) * hidden_size + i]) * sn;
            }
        }
    }

    // Reduce each row's dot product
    #pragma unroll
    for (int r = 0; r < RPB; r++) {
        if (r >= rows_this_block) break;

        float val = warp_reduce_sum_opag(acc[r]);
        if (lane_id == 0) s_warp[warp_id] = val;
        __syncthreads();

        if (warp_id == 0) {
            float v = (lane_id < NUM_WARPS) ? s_warp[lane_id] : 0.0f;
            v = warp_reduce_sum_opag(v);
            if (lane_id == 0) {
                gate_up_out[block_row_base + r] = __float2half(v);
            }
        }
        __syncthreads();
    }
}

// --------------------------------------------------------------------------
// FP8 variant: both o_weight and gateup_weight are E4M3 with per-row scales
// --------------------------------------------------------------------------
extern "C"
__global__ void __launch_bounds__(THREADS)
fused_cute_oproj_add_norm_gateup_fp8_gemv(
    __half* __restrict__ gate_up_out,
    __half* __restrict__ residual_out,
    const __half* __restrict__ attn_out,
    const unsigned char* __restrict__ o_weight,
    const __half* __restrict__ o_weight_scale,
    const __half* __restrict__ residual,
    const __half* __restrict__ norm_weight,
    const unsigned char* __restrict__ gateup_weight,
    const __half* __restrict__ gateup_weight_scale,
    float eps,
    int q_dim,
    int hidden_size,
    int gate_up_dim
) {
    const int block_row_base = blockIdx.x * RPB;
    if (block_row_base >= gate_up_dim) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int NUM_WARPS = THREADS / 32;

    extern __shared__ float smem[];
    float* s_hidden = smem;
    float* s_warp   = smem + hidden_size;

    // ---- Phase 1: O-projection GEMV with FP8 weights ----
    const int q4 = q_dim / 4;

    for (int h = 0; h < hidden_size; h++) {
        const unsigned char* ow_row = o_weight + (long long)h * q_dim;
        float o_sc = __half2float(o_weight_scale[h]);
        float acc = 0.0f;

        for (int i = tid; i < q4; i += THREADS) {
            int base = i * 4;
            unsigned int packed = *reinterpret_cast<const unsigned int*>(ow_row + base);
            unsigned char b0 = packed & 0xFF;
            unsigned char b1 = (packed >> 8) & 0xFF;
            unsigned char b2 = (packed >> 16) & 0xFF;
            unsigned char b3 = (packed >> 24) & 0xFF;
            float w0 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b0)) * o_sc;
            float w1 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b1)) * o_sc;
            float w2 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b2)) * o_sc;
            float w3 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b3)) * o_sc;
            acc += __half2float(attn_out[base])     * w0
                 + __half2float(attn_out[base + 1]) * w1
                 + __half2float(attn_out[base + 2]) * w2
                 + __half2float(attn_out[base + 3]) * w3;
        }
        for (int i = q4 * 4 + tid; i < q_dim; i += THREADS) {
            unsigned char b = ow_row[i];
            float w = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b)) * o_sc;
            acc += __half2float(attn_out[i]) * w;
        }

        acc = warp_reduce_sum_opag(acc);
        if (lane_id == 0) s_warp[warp_id] = acc;
        __syncthreads();

        if (warp_id == 0) {
            float val = (lane_id < NUM_WARPS) ? s_warp[lane_id] : 0.0f;
            val = warp_reduce_sum_opag(val);
            if (lane_id == 0) {
                s_hidden[h] = val;
            }
        }
        if (h + 1 < hidden_size) __syncthreads();
    }
    __syncthreads();

    // ---- Phase 2: Residual add + RMSNorm (unchanged) ----
    const int h2 = hidden_size / 2;
    const half2* res2 = (const half2*)residual;

    float local_ss = 0.0f;
    for (int i = tid; i < h2; i += THREADS) {
        half2 r = res2[i];
        float v0 = s_hidden[i * 2]     + __half2float(r.x);
        float v1 = s_hidden[i * 2 + 1] + __half2float(r.y);
        s_hidden[i * 2]     = v0;
        s_hidden[i * 2 + 1] = v1;
        local_ss += v0 * v0 + v1 * v1;
    }
    if ((hidden_size & 1) && tid == 0) {
        int last = hidden_size - 1;
        float v = s_hidden[last] + __half2float(residual[last]);
        s_hidden[last] = v;
        local_ss += v * v;
    }

    local_ss = warp_reduce_sum_opag(local_ss);
    if (lane_id == 0) s_warp[warp_id] = local_ss;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_warp[lane_id] : 0.0f;
        val = warp_reduce_sum_opag(val);
        if (lane_id == 0) {
            s_warp[0] = rsqrtf(val / (float)hidden_size + eps);
        }
    }
    __syncthreads();

    float rms_scale = s_warp[0];

    if (blockIdx.x == 0) {
        for (int i = tid; i < hidden_size; i += THREADS) {
            residual_out[i] = __float2half(s_hidden[i]);
        }
    }

    for (int i = tid; i < hidden_size; i += THREADS) {
        s_hidden[i] = s_hidden[i] * __half2float(norm_weight[i]) * rms_scale;
    }
    __syncthreads();

    // ---- Phase 3: gate_up GEMV with FP8 weights -- warp-per-row ----
    {
        const int row = block_row_base + warp_id;
        if (row < gate_up_dim) {
            const unsigned char* w_row = gateup_weight + (long long)row * hidden_size;
            float gu_sc = __half2float(gateup_weight_scale[row]);
            float acc = 0.0f;
            const int h4 = hidden_size / 4;
            for (int i = lane_id; i < h4; i += 32) {
                int base = i * 4;
                unsigned int packed = *reinterpret_cast<const unsigned int*>(w_row + base);
                unsigned char b0 = packed & 0xFF;
                unsigned char b1 = (packed >> 8) & 0xFF;
                unsigned char b2 = (packed >> 16) & 0xFF;
                unsigned char b3 = (packed >> 24) & 0xFF;
                float w0 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b0)) * gu_sc;
                float w1 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b1)) * gu_sc;
                float w2 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b2)) * gu_sc;
                float w3 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b3)) * gu_sc;
                acc += w0 * s_hidden[base]
                     + w1 * s_hidden[base + 1]
                     + w2 * s_hidden[base + 2]
                     + w3 * s_hidden[base + 3];
            }
            for (int i = h4 * 4 + lane_id; i < hidden_size; i += 32) {
                unsigned char b = w_row[i];
                float w = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b)) * gu_sc;
                acc += w * s_hidden[i];
            }
            acc = warp_reduce_sum_opag(acc);
            if (lane_id == 0) {
                gate_up_out[row] = __float2half(acc);
            }
        }
    }
}
