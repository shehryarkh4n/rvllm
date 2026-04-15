// Fused LM-head matvec + argmax for single-token decode (M=1).
// Eliminates materializing the full [vocab_size] logits tensor.
//
// Pass 1 (fused_lm_head_argmax_kernel):
//   Grid:  (ceil(vocab_size / BLOCK_SIZE), 1, 1)
//   Block: (BLOCK_SIZE, 1, 1)    -- BLOCK_SIZE = 256
//   Shared: hidden_size * sizeof(float) bytes
//
//   Each thread computes dot(weight_row, hidden_state) for one vocab row,
//   then the block reduces to find the local argmax. Partial results are
//   written to partial_val[blockIdx.x] and partial_idx[blockIdx.x].
//
// Pass 2 (fused_lm_head_argmax_reduce_kernel):
//   Grid:  (1, 1, 1)
//   Block: (min(num_blocks, 1024), 1, 1)
//   Shared: none (static arrays)
//
//   Reduces partial results to a single winning token ID.

#include <float.h>

extern "C"
__global__ void fused_lm_head_argmax_kernel(
    const float* __restrict__ weight,        // [vocab_size, hidden_size] row-major
    const float* __restrict__ hidden_state,  // [hidden_size]
    float* __restrict__ partial_val,         // [num_blocks]
    int*   __restrict__ partial_idx,         // [num_blocks]
    int vocab_size,
    int hidden_size
) {
    const int tid = threadIdx.x;
    const int row = blockIdx.x * blockDim.x + tid;

    // Load hidden_state into shared memory (one cooperative load per block)
    extern __shared__ float s_hidden[];
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        s_hidden[i] = hidden_state[i];
    }
    __syncthreads();

    // Each thread computes dot product for its vocab row
    float dot = -FLT_MAX;
    int   my_idx = row;

    if (row < vocab_size) {
        const float* w = weight + (long long)row * hidden_size;

        // Vectorized loads (float4 = 16 bytes)
        const float4* w4 = reinterpret_cast<const float4*>(w);
        const float4* h4 = reinterpret_cast<const float4*>(s_hidden);
        int vec_len = hidden_size >> 2;

        float acc = 0.0f;
        for (int i = 0; i < vec_len; i++) {
            float4 wv = w4[i];
            float4 hv = h4[i];
            acc += wv.x * hv.x + wv.y * hv.y + wv.z * hv.z + wv.w * hv.w;
        }
        // Tail elements (hidden_size not divisible by 4)
        for (int i = vec_len << 2; i < hidden_size; i++) {
            acc += w[i] * s_hidden[i];
        }
        dot = acc;
    }

    // Intra-warp argmax reduction via shuffles
    unsigned mask = 0xFFFFFFFF;
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val = __shfl_down_sync(mask, dot, offset);
        int   other_idx = __shfl_down_sync(mask, my_idx, offset);
        if (other_val > dot) {
            dot = other_val;
            my_idx = other_idx;
        }
    }

    // Cross-warp reduction via shared memory (reuse s_hidden tail area)
    // We need blockDim.x / 32 entries. Max 256/32 = 8 warps.
    const int WARPS_MAX = 32;  // supports up to 1024 threads
    __shared__ float  s_wval[WARPS_MAX];
    __shared__ int    s_widx[WARPS_MAX];

    int warp_id = tid >> 5;
    int lane    = tid & 31;
    int num_warps = (blockDim.x + 31) >> 5;

    if (lane == 0) {
        s_wval[warp_id] = dot;
        s_widx[warp_id] = my_idx;
    }
    __syncthreads();

    // First warp reduces across warp leaders
    if (warp_id == 0 && lane < num_warps) {
        dot    = s_wval[lane];
        my_idx = s_widx[lane];

        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_val = __shfl_down_sync(mask, dot, offset);
            int   other_idx = __shfl_down_sync(mask, my_idx, offset);
            if (other_val > dot) {
                dot = other_val;
                my_idx = other_idx;
            }
        }
    }

    if (tid == 0) {
        partial_val[blockIdx.x] = dot;
        partial_idx[blockIdx.x] = my_idx;
    }
}

// Pass 2: reduce partial results from all blocks into a single token ID.
extern "C"
__global__ void fused_lm_head_argmax_reduce_kernel(
    const float* __restrict__ partial_val,   // [num_blocks]
    const int*   __restrict__ partial_idx,   // [num_blocks]
    int*         __restrict__ output_token,  // [1]
    int num_blocks
) {
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    __shared__ float s_val[1024];
    __shared__ int   s_idx[1024];

    float local_max = -FLT_MAX;
    int   local_idx = 0;
    for (int i = tid; i < num_blocks; i += stride) {
        float v = partial_val[i];
        if (v > local_max) {
            local_max = v;
            local_idx = partial_idx[i];
        }
    }
    s_val[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();

    // Tree reduction
    int n = blockDim.x;
    for (int s = n / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < n) {
            if (s_val[tid + s] > s_val[tid]) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        if (s * 2 < n && tid == 0) {
            if (s_val[s * 2] > s_val[0]) {
                s_val[0] = s_val[s * 2];
                s_idx[0] = s_idx[s * 2];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output_token[0] = s_idx[0];
    }
}
