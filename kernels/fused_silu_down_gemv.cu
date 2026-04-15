// Fused SiLU*Mul + Down-projection GEMV (f16 I/O, f32 accumulation).
//
// Each block computes RPB=8 output rows. With 256 threads and 8 rows,
// each row is handled by one warp (32 threads). This gives 8x fewer blocks
// than the 1-row-per-block variant, improving occupancy and L2 reuse of
// gate/up vectors across the 8 rows sharing a block.
//
//   output[r] = sum_i( silu(gate[i]) * up[i] * weight[r, i] )
//
// Launch config:
//   Grid:  ((hidden_size + 7) / 8, 1, 1)
//   Block: (256, 1, 1)
//   Shared memory: RPB * sizeof(float) = 32 bytes

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define CUTE_SILU_THREADS 256
#define CUTE_SILU_RPB 8
#define CUTE_SILU_WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_sum_csg(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float silu_f32_csg(float x) {
    return x / (1.0f + expf(-x));
}

extern "C"
__global__ void __launch_bounds__(CUTE_SILU_THREADS)
fused_cute_silu_down_gemv(
    __half* __restrict__ output,         // [hidden_size]
    const __half* __restrict__ gate,     // [intermediate_size]
    const __half* __restrict__ up,       // [intermediate_size]
    const __half* __restrict__ weight,   // [hidden_size, intermediate_size]
    int hidden_size,
    int intermediate_size
) {
    // 8 rows per block, 1 warp per row
    const int base_row = blockIdx.x * CUTE_SILU_RPB;
    const int warp_id = threadIdx.x / CUTE_SILU_WARP_SIZE;
    const int lane_id = threadIdx.x % CUTE_SILU_WARP_SIZE;
    const int row = base_row + warp_id;

    if (row >= hidden_size) return;

    const __half* w_row = weight + (long long)row * intermediate_size;
    float acc = 0.0f;

    // 128-bit vectorized loads: 8 f16 per load, 4x fewer instructions than half2
    const int k8 = intermediate_size / 8;
    const int4* gate4 = (const int4*)gate;
    const int4* up4   = (const int4*)up;
    const int4* w4    = (const int4*)w_row;

    #pragma unroll 4
    for (int i = lane_id; i < k8; i += CUTE_SILU_WARP_SIZE) {
        int4 gv = gate4[i];
        int4 uv = up4[i];
        int4 wv = w4[i];

        half2 g01 = *reinterpret_cast<half2*>(&gv.x);
        half2 g23 = *reinterpret_cast<half2*>(&gv.y);
        half2 g45 = *reinterpret_cast<half2*>(&gv.z);
        half2 g67 = *reinterpret_cast<half2*>(&gv.w);
        half2 u01 = *reinterpret_cast<half2*>(&uv.x);
        half2 u23 = *reinterpret_cast<half2*>(&uv.y);
        half2 u45 = *reinterpret_cast<half2*>(&uv.z);
        half2 u67 = *reinterpret_cast<half2*>(&uv.w);
        half2 w01 = *reinterpret_cast<half2*>(&wv.x);
        half2 w23 = *reinterpret_cast<half2*>(&wv.y);
        half2 w45 = *reinterpret_cast<half2*>(&wv.z);
        half2 w67 = *reinterpret_cast<half2*>(&wv.w);

        acc += silu_f32_csg(__half2float(g01.x)) * __half2float(u01.x) * __half2float(w01.x);
        acc += silu_f32_csg(__half2float(g01.y)) * __half2float(u01.y) * __half2float(w01.y);
        acc += silu_f32_csg(__half2float(g23.x)) * __half2float(u23.x) * __half2float(w23.x);
        acc += silu_f32_csg(__half2float(g23.y)) * __half2float(u23.y) * __half2float(w23.y);
        acc += silu_f32_csg(__half2float(g45.x)) * __half2float(u45.x) * __half2float(w45.x);
        acc += silu_f32_csg(__half2float(g45.y)) * __half2float(u45.y) * __half2float(w45.y);
        acc += silu_f32_csg(__half2float(g67.x)) * __half2float(u67.x) * __half2float(w67.x);
        acc += silu_f32_csg(__half2float(g67.y)) * __half2float(u67.y) * __half2float(w67.y);
    }

    // Handle remainder (intermediate_size not multiple of 8)
    for (int i = k8 * 8 + lane_id; i < intermediate_size; i += CUTE_SILU_WARP_SIZE) {
        float g = __half2float(gate[i]);
        acc += silu_f32_csg(g) * __half2float(up[i]) * __half2float(w_row[i]);
    }

    // Warp-level reduction (no shared memory needed within a warp)
    acc = warp_reduce_sum_csg(acc);

    // Lane 0 of each warp writes the result
    if (lane_id == 0) {
        output[row] = __float2half(acc);
    }
}

// --------------------------------------------------------------------------
// FP8 variant: weights stored as E4M3 bytes with per-row scale
// --------------------------------------------------------------------------
extern "C"
__global__ void __launch_bounds__(CUTE_SILU_THREADS)
fused_cute_silu_down_fp8_gemv(
    __half* __restrict__ output,
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    const unsigned char* __restrict__ weight,
    const __half* __restrict__ weight_scale,
    int hidden_size,
    int intermediate_size
) {
    const int base_row = blockIdx.x * CUTE_SILU_RPB;
    const int warp_id = threadIdx.x / CUTE_SILU_WARP_SIZE;
    const int lane_id = threadIdx.x % CUTE_SILU_WARP_SIZE;
    const int row = base_row + warp_id;

    if (row >= hidden_size) return;

    const unsigned char* w_row = weight + (long long)row * intermediate_size;
    float row_sc = __half2float(weight_scale[row]);
    float acc = 0.0f;

    // 4-wide byte loads for coalescing
    const int k4 = intermediate_size / 4;
    const half2* gate2 = (const half2*)gate;
    const half2* up2   = (const half2*)up;

    for (int i = lane_id; i < k4; i += CUTE_SILU_WARP_SIZE) {
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

        // base/2 gives the half2 index for the pair (base, base+1)
        half2 g01 = gate2[base / 2];
        half2 g23 = gate2[base / 2 + 1];
        half2 u01 = up2[base / 2];
        half2 u23 = up2[base / 2 + 1];

        float g0 = __half2float(g01.x);
        float g1 = __half2float(g01.y);
        float g2 = __half2float(g23.x);
        float g3 = __half2float(g23.y);
        float u0 = __half2float(u01.x);
        float u1 = __half2float(u01.y);
        float u2 = __half2float(u23.x);
        float u3 = __half2float(u23.y);

        acc += silu_f32_csg(g0) * u0 * w0;
        acc += silu_f32_csg(g1) * u1 * w1;
        acc += silu_f32_csg(g2) * u2 * w2;
        acc += silu_f32_csg(g3) * u3 * w3;
    }

    // Handle remaining elements
    for (int i = k4 * 4 + lane_id; i < intermediate_size; i += CUTE_SILU_WARP_SIZE) {
        unsigned char b = w_row[i];
        float w = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b)) * row_sc;
        float g = __half2float(gate[i]);
        acc += silu_f32_csg(g) * __half2float(up[i]) * w;
    }

    acc = warp_reduce_sum_csg(acc);

    if (lane_id == 0) {
        output[row] = __float2half(acc);
    }
}
