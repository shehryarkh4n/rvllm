// Fused RoPE + KV Cache Write kernel.
// Applies rotary position embeddings to Q and K in-place,
// then writes K and V to the paged KV cache in a single kernel.
// Saves 1 kernel launch vs separate RoPE + cache_write.
//
// Grid: (num_tokens, max(num_heads, num_kv_heads), 1)
// Block: (half_dim, 1, 1) where half_dim = head_dim / 2

#include <cuda_fp16.h>

extern "C"
__global__ void fused_rope_cache_f16_kernel(
    __half* __restrict__ q,           // [num_tokens, num_heads * head_dim] -- RoPE applied in-place
    __half* __restrict__ k,           // [num_tokens, num_kv_heads * head_dim] -- RoPE applied in-place
    const __half* __restrict__ v,     // [num_tokens, num_kv_heads * head_dim] -- read-only
    __half* __restrict__ key_cache,   // [num_blocks, block_size, num_kv_heads, head_dim]
    __half* __restrict__ value_cache, // [num_blocks, block_size, num_kv_heads, head_dim]
    const float* __restrict__ cos_table, // [max_pos, half_dim]
    const float* __restrict__ sin_table, // [max_pos, half_dim]
    const int* __restrict__ positions,   // [num_tokens]
    const int* __restrict__ slot_mapping, // [num_tokens]
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    const int token_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int half_dim = head_dim / 2;
    const int tid = threadIdx.x;
    if (tid >= half_dim) return;

    const int pos = positions[token_idx];
    const float cos_val = cos_table[pos * half_dim + tid];
    const float sin_val = sin_table[pos * half_dim + tid];

    // Apply RoPE to Q (all heads)
    if (head_idx < num_heads) {
        int q_base = (token_idx * num_heads + head_idx) * head_dim;
        float q0 = __half2float(q[q_base + 2 * tid]);
        float q1 = __half2float(q[q_base + 2 * tid + 1]);
        q[q_base + 2 * tid]     = __float2half(q0 * cos_val - q1 * sin_val);
        q[q_base + 2 * tid + 1] = __float2half(q0 * sin_val + q1 * cos_val);
    }

    // Apply RoPE to K (kv_heads only) + write to KV cache
    if (head_idx < num_kv_heads) {
        int k_base = (token_idx * num_kv_heads + head_idx) * head_dim;
        float k0 = __half2float(k[k_base + 2 * tid]);
        float k1 = __half2float(k[k_base + 2 * tid + 1]);
        float k0_rot = k0 * cos_val - k1 * sin_val;
        float k1_rot = k0 * sin_val + k1 * cos_val;
        k[k_base + 2 * tid]     = __float2half(k0_rot);
        k[k_base + 2 * tid + 1] = __float2half(k1_rot);

        // Write rotated K to cache
        int slot = slot_mapping[token_idx];
        if (slot >= 0) {
            int cache_offset = (slot * num_kv_heads + head_idx) * head_dim;
            key_cache[cache_offset + 2 * tid]     = __float2half(k0_rot);
            key_cache[cache_offset + 2 * tid + 1] = __float2half(k1_rot);

            // Write V to cache (no RoPE on V)
            int v_base = (token_idx * num_kv_heads + head_idx) * head_dim;
            value_cache[cache_offset + 2 * tid]     = v[v_base + 2 * tid];
            value_cache[cache_offset + 2 * tid + 1] = v[v_base + 2 * tid + 1];
        }
    }
}
