// Fused LM-head matvec + argmax for single-token decode (M=1), f16 weights.
// Weight matrix is __half, hidden state is f32, accumulation in f32.
//
// Same two-pass structure as the f32 variant. Pass 2 reuses the f32 reduce kernel.

#include <cuda_fp16.h>
#include <float.h>

extern "C"
__global__ void fused_lm_head_argmax_f16_kernel(
    const __half* __restrict__ weight,       // [vocab_size, hidden_size] row-major, f16
    const float*  __restrict__ hidden_state, // [hidden_size] f32
    float* __restrict__ partial_val,         // [num_blocks]
    int*   __restrict__ partial_idx,         // [num_blocks]
    int vocab_size,
    int hidden_size
) {
    const int tid = threadIdx.x;
    const int row = blockIdx.x * blockDim.x + tid;

    // Load hidden_state into shared memory as f32
    extern __shared__ float s_hidden[];
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        s_hidden[i] = hidden_state[i];
    }
    __syncthreads();

    float dot = -FLT_MAX;
    int   my_idx = row;

    if (row < vocab_size) {
        const __half* w = weight + (long long)row * hidden_size;

        // Vectorized loads: half2 = 2 x f16 in one 32-bit load
        const half2* w2 = reinterpret_cast<const half2*>(w);
        int vec_len = hidden_size >> 1;

        float acc = 0.0f;
        for (int i = 0; i < vec_len; i++) {
            half2 wv = w2[i];
            int base = i << 1;
            acc += __half2float(wv.x) * s_hidden[base]
                 + __half2float(wv.y) * s_hidden[base + 1];
        }
        // Tail element (odd hidden_size)
        if ((hidden_size & 1) && (hidden_size > 0)) {
            int last = hidden_size - 1;
            acc += __half2float(w[last]) * s_hidden[last];
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

    // Cross-warp reduction
    const int WARPS_MAX = 32;
    __shared__ float s_wval[WARPS_MAX];
    __shared__ int   s_widx[WARPS_MAX];

    int warp_id = tid >> 5;
    int lane    = tid & 31;
    int num_warps = (blockDim.x + 31) >> 5;

    if (lane == 0) {
        s_wval[warp_id] = dot;
        s_widx[warp_id] = my_idx;
    }
    __syncthreads();

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
