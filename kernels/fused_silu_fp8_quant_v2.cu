// Fused SiLU(gate)*up + per-TENSOR FP8 E4M3 quantization for SM90.
// v2: uses atomicMax + spin counter for cross-block global absmax.
// Scratch: output_scales must have (num_tokens + 2) floats, last 2 zeroed.

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define FP8_E4M3_MAX 448.0f
#define WARPS_MAX 32
#define VEC_SIZE 8
#define MAX_VECS_PER_THREAD 4

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__ float block_reduce_max(float val, float* smem) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    val = warp_reduce_max(val);
    if (lane_id == 0) smem[warp_id] = val;
    __syncthreads();
    int num_warps = (blockDim.x + 31) / 32;
    val = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
    if (warp_id == 0) val = warp_reduce_max(val);
    return val;
}

__device__ __forceinline__ float grid_absmax_barrier(
    float local_absmax, float* scales, int num_rows
) {
    if (threadIdx.x == 0) {
        atomicMax((unsigned int*)&scales[num_rows],
                  __float_as_uint(local_absmax));
        __threadfence();
        atomicAdd((unsigned int*)&scales[num_rows + 1], 1u);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        while (atomicAdd((unsigned int*)&scales[num_rows + 1], 0u)
               < (unsigned int)num_rows) {
            __nanosleep(32);
        }
    }
    __syncthreads();
    return __uint_as_float(
        atomicAdd((unsigned int*)&scales[num_rows], 0u));
}

extern "C" __global__ void __launch_bounds__(1024)
fused_silu_mul_fp8_quant_kernel(
    __nv_fp8_storage_t* __restrict__ output_fp8,
    float*              __restrict__ output_scales,
    const __half*       __restrict__ gate_up,
    int intermediate_size
) {
    const int row = blockIdx.x;
    const int num_rows = gridDim.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int n_vecs = intermediate_size / VEC_SIZE;

    const uint4* gate_vec = reinterpret_cast<const uint4*>(
        gate_up + row * 2 * intermediate_size);
    const uint4* up_vec = reinterpret_cast<const uint4*>(
        gate_up + row * 2 * intermediate_size + intermediate_size);

    __shared__ float smem[WARPS_MAX];

    float cached[MAX_VECS_PER_THREAD * VEC_SIZE];

    // Pass 1: compute SiLU(gate)*up, find per-row absmax
    float local_max = 0.0f;
    int vec_idx = 0;
    for (int i = tid; i < n_vecs; i += stride, vec_idx++) {
        uint4 gv = gate_vec[i];
        uint4 uv = up_vec[i];
        const __half* g = reinterpret_cast<const __half*>(&gv);
        const __half* u = reinterpret_cast<const __half*>(&uv);

        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            float gf = __half2float(g[j]);
            float uf = __half2float(u[j]);
            float silu_g = gf / (1.0f + expf(-gf));
            float v = silu_g * uf;
            cached[vec_idx * VEC_SIZE + j] = v;
            local_max = fmaxf(local_max, fabsf(v));
        }
    }

    float row_absmax = block_reduce_max(local_max, smem);
    if (threadIdx.x == 0) smem[0] = row_absmax;
    __syncthreads();
    row_absmax = smem[0];

    // Cross-block per-tensor absmax
    float global_absmax = grid_absmax_barrier(row_absmax, output_scales, num_rows);

    float scale = global_absmax / FP8_E4M3_MAX;
    scale = fmaxf(scale, 1e-12f);
    if (row == 0 && tid == 0) output_scales[0] = scale;
    float inv_scale = 1.0f / scale;

    // Pass 2: quantize from register cache
    uint2* out_vec = reinterpret_cast<uint2*>(output_fp8 + row * intermediate_size);
    vec_idx = 0;
    for (int i = tid; i < n_vecs; i += stride, vec_idx++) {
        __nv_fp8_storage_t fp8[VEC_SIZE];
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            fp8[j] = __nv_cvt_float_to_fp8(
                cached[vec_idx * VEC_SIZE + j] * inv_scale,
                __NV_SATFINITE, __NV_E4M3);
        }
        out_vec[i] = *reinterpret_cast<const uint2*>(fp8);
    }
}
