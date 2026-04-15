// QKV transpose kernel: rearrange fused GEMM output from per-row interleaved
// to contiguous-per-projection layout.
//
// Input:  src[N, qkv_dim] where each row is [Q_i(q_dim), K_i(kv_dim), V_i(kv_dim)]
// Output: dst[N * qkv_dim] laid out as [Q_all(N*q_dim) | K_all(N*kv_dim) | V_all(N*kv_dim)]
//
// This enables using a single fused GEMM for QKV projection at any batch size,
// then splitting Q/K/V as contiguous slices for RoPE and attention.
//
// Grid: (ceil(N * qkv_dim / 256), 1, 1)  Block: (256, 1, 1)

#include <cuda_fp16.h>

extern "C"
__global__ void qkv_transpose_f16_kernel(
    __half* __restrict__ dst,        // [N*qkv_dim] as [all_Q | all_K | all_V]
    const __half* __restrict__ src,  // [N, qkv_dim] row-interleaved
    int N,
    int q_dim,
    int kv_dim
) {
    const int qkv_dim = q_dim + kv_dim + kv_dim;
    const int total = N * qkv_dim;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Which token and which element within that token's QKV?
    const int token = idx / qkv_dim;
    const int elem = idx % qkv_dim;

    // Source: row-interleaved [token * qkv_dim + elem]
    __half val = src[token * qkv_dim + elem];

    // Destination: contiguous per-projection
    int dst_idx;
    if (elem < q_dim) {
        // Q section: dst[token * q_dim + elem]
        dst_idx = token * q_dim + elem;
    } else if (elem < q_dim + kv_dim) {
        // K section: dst[N * q_dim + token * kv_dim + (elem - q_dim)]
        dst_idx = N * q_dim + token * kv_dim + (elem - q_dim);
    } else {
        // V section: dst[N * q_dim + N * kv_dim + token * kv_dim + (elem - q_dim - kv_dim)]
        dst_idx = N * q_dim + N * kv_dim + token * kv_dim + (elem - q_dim - kv_dim);
    }

    dst[dst_idx] = val;
}

// Same transpose for gate+up: [N, 2*I] interleaved -> [all_gate(N*I) | all_up(N*I)]
extern "C"
__global__ void gateup_transpose_f16_kernel(
    __half* __restrict__ dst,
    const __half* __restrict__ src,
    int N,
    int intermediate_size
) {
    const int gate_up_dim = intermediate_size * 2;
    const int total = N * gate_up_dim;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int token = idx / gate_up_dim;
    const int elem = idx % gate_up_dim;

    __half val = src[token * gate_up_dim + elem];

    int dst_idx;
    if (elem < intermediate_size) {
        dst_idx = token * intermediate_size + elem;
    } else {
        dst_idx = N * intermediate_size + token * intermediate_size + (elem - intermediate_size);
    }

    dst[dst_idx] = val;
}
