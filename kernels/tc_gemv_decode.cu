// Tensor-core GEMV for M=1 decode with cp.async double-buffered weight prefetch.
//
// y[n] = weight[n, k] @ x[k]   (f16 I/O, f32 accumulation via tensor cores)
//
// Combines:
//   - tma_gemv_fp16.cu: cp.async double-buffering for weight prefetch
//   - wgmma_gemv.cu:    m16n8k16 tensor core MMA for dot products
//
// Strategy per block:
//   256 threads = 8 warps. Each warp owns 8 output rows (via two m16n8k16 tiles
//   that each produce 8 results). Weight tiles [8, TILE_K] are double-buffered
//   in shared memory via cp.async. Input vector is cached in shared memory with
//   int4 (128-bit) loads. Each warp issues m16n8k16 MMA instructions per K-step.
//
// Why tensor cores beat scalar on GEMV:
//   Scalar: 2 FMA/thread/cycle (half2 FMA) = 64 FMA/warp/cycle
//   Tensor: m16n8k16 = 2048 FMA/warp/instruction (~4 cycles) = 512 FMA/warp/cycle
//   ~8x more compute throughput. GEMV is memory-bound so this doesn't directly
//   help bandwidth, but it frees up CUDA cores for fused epilogues and reduces
//   the number of K-iterations needed before warp reduction (no reduction needed --
//   the accumulator lives in registers across all K-steps).
//
// m16n8k16 tensor core op (Ampere/Hopper, sm_80+):
//   A[16, 16] row-major (f16) x B[16, 8] col-major (f16) -> C[16, 8] (f32)
//   For GEMV: A = broadcast(input[k:k+16]) to 16 rows, B = weight[8 rows, 16 cols]^T
//   Row 0 of C contains 8 dot-product partial sums.
//
// Launch config:
//   Grid:  ((out_dim + BLOCK_N - 1) / BLOCK_N, 1, 1)  where BLOCK_N = 64
//   Block: (256, 1, 1)
//   Shared mem: align128(hidden_size * 2) + 2 * 8 * 8 * 128 * 2 bytes
//             = e.g. hidden=3584: 7168 -> 7168 + 2 * 8 * 1024 * 2 = 7168 + 32768 = 39936 bytes

#include <cuda_fp16.h>
#include <cstdint>

// Requires: hidden_size multiple of 16 (for MMA k-dim alignment)

// ---- Config ----
#define TC_THREADS 256
#define TC_WARPS   (TC_THREADS / 32)   // 8
#define TC_RPW     8                    // rows per warp (output rows handled by one warp)
#define BLOCK_N    (TC_WARPS * TC_RPW)  // 64 output rows per block

// cp.async: copy 16 bytes from global to shared
__device__ __forceinline__ void cp_async_16b(void* smem, const void* gmem) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;"
        :: "r"(addr), "l"(gmem) : "memory"
    );
}

__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_group_0() {
    asm volatile("cp.async.wait_group 0;" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_group_1() {
    asm volatile("cp.async.wait_group 1;" ::: "memory");
}

// m16n8k16 tensor core MMA (f16 x f16 -> f32)
// A: 16x16 row-major in registers (4 half2 fragments per thread)
// B: 16x8  col-major in registers (2 half2 fragments per thread)
// C/D: 16x8 in registers (4 float per thread)
__device__ __forceinline__ void mma_m16n8k16_f16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3)
    );
}

// Load A fragment for m16n8k16 from shared memory (row-major, 16x16).
// Thread mapping for A fragment (row-major):
//   Thread t in warp owns rows [t/4, t/4+8] (for t%4 selecting the k-column group).
//   Fragment layout: a0 = half2(A[row0, col0..1]), a1 = half2(A[row0, col2..3]),
//                    a2 = half2(A[row8, col0..1]), a3 = half2(A[row8, col2..3])
//   where row0 = t/4, row8 = t/4 + 8, col = (t%4)*4 for a0/a2, (t%4)*4+2 for a1/a3
//
// For GEMV broadcast: all 16 rows of A are identical (= input_vector[k:k+16]).
// So we just replicate the same 16 f16 values to all rows.
__device__ __forceinline__ void load_a_broadcast(
    uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3,
    const __half* __restrict__ s_vec,  // 16 f16 values in shared memory
    int lane_id)
{
    // For m16n8k16 row-major A:
    //   Thread lane_id owns:
    //     a0 = half2(A[row0, col], A[row0, col+1]) where row0 = lane_id/4, col = (lane_id%4)*2
    //     a1 = half2(A[row0, col+8], A[row0, col+9])
    //     a2 = half2(A[row8, col], A[row8, col+1])
    //     a3 = half2(A[row8, col+8], A[row8, col+9])
    // Since all rows are identical (broadcast), row index doesn't matter.
    int col_base = (lane_id % 4) * 2;

    // a0: columns col_base, col_base+1
    // a1: columns col_base+8, col_base+9
    // a2: same as a0 (row+8 = same data for broadcast)
    // a3: same as a1
    half2 v01, v89;
    v01 = *reinterpret_cast<const half2*>(&s_vec[col_base]);
    v89 = *reinterpret_cast<const half2*>(&s_vec[col_base + 8]);

    a0 = *reinterpret_cast<const uint32_t*>(&v01);
    a1 = *reinterpret_cast<const uint32_t*>(&v89);
    a2 = a0;  // broadcast: row+8 = row+0
    a3 = a1;
}

// m16n8k16 f32 accumulator output layout (per PTX ISA):
//   Thread t (0..31):
//     groupID = t / 4  (0..7)
//     tid_in_group = t % 4  (0..3)
//     d0 = D[groupID,     tid_in_group]       (rows 0..7,  cols 0..3)
//     d1 = D[groupID + 8, tid_in_group]       (rows 8..15, cols 0..3)
//     d2 = D[groupID,     tid_in_group + 4]   (rows 0..7,  cols 4..7)
//     d3 = D[groupID + 8, tid_in_group + 4]   (rows 8..15, cols 4..7)
//   Total: 128 floats = 16 * 8. Correct.
//
// For GEMV: we want row 0 of D = D[0, 0..7].
//   D[0, col] for col=0..3: groupID=0 -> threads 0..3, register d0
//   D[0, col+4]: same threads, register d2
// So threads 0..3 hold all 8 output values in d0 and d2.

extern "C"
__global__ void __launch_bounds__(TC_THREADS, 2)
tc_gemv_decode_f16(
    __half* __restrict__ output,       // [out_dim]
    const __half* __restrict__ input,  // [hidden_size]
    const __half* __restrict__ weight, // [out_dim, hidden_size] row-major
    int out_dim,
    int hidden_size
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // This block handles BLOCK_N = 64 output rows
    const int block_base = blockIdx.x * BLOCK_N;
    // Each warp handles TC_RPW = 8 output rows
    const int warp_base = block_base + warp_id * TC_RPW;

    // Shared memory layout:
    //   [0, align128(hidden*2)):  input vector (f16)
    //   [align128, ...):          double-buffered weight tiles [2][WARPS][RPW][TILE_KK]
    //   TILE_KK=128: 2 * 8 * 8 * 128 * 2 = 32768 bytes for weight buffers

    extern __shared__ char smem_raw[];
    __half* s_input = (__half*)smem_raw;
    // Weight tiles start after input vector (aligned to 128 bytes)
    const int input_bytes = ((hidden_size * sizeof(__half)) + 127) & ~127;
    __half* s_weight = (__half*)(smem_raw + input_bytes);
    // s_weight layout: [2][TC_WARPS][TC_RPW][tile_k] but we use a flat offset

    // Use TILE_K = 128 for the double-buffer tile width
    const int TILE_KK = 128;
    // Per warp slot size in f16 elements
    const int warp_tile_elems = TC_RPW * TILE_KK;  // 8 * 128 = 1024
    const int buf_elems = TC_WARPS * warp_tile_elems;  // 8 * 1024 = 8192

    // ---- Phase 1: Load input vector into shared memory (all threads) ----
    // Use int4 (128-bit) loads for maximum bandwidth
    const int elems_per_int4 = 8;  // 128 bits / 16 bits per half = 8
    const int n_int4 = hidden_size / elems_per_int4;
    const int4* input_int4 = (const int4*)input;
    int4* s_input_int4 = (int4*)s_input;
    for (int i = tid; i < n_int4; i += TC_THREADS) {
        s_input_int4[i] = input_int4[i];
    }
    // Handle remainder (hidden_size not multiple of 8)
    for (int i = n_int4 * elems_per_int4 + tid; i < hidden_size; i += TC_THREADS) {
        s_input[i] = input[i];
    }
    __syncthreads();

    if (warp_base >= out_dim) return;
    const int valid_rows = min(TC_RPW, out_dim - warp_base);

    // ---- Phase 2: Double-buffered weight prefetch + MMA compute ----

    const int num_tiles = (hidden_size + TILE_KK - 1) / TILE_KK;

    // Prefetch tile 0 into buffer 0
    {
        const int tile_end = min(TILE_KK, hidden_size);
        __half* dst = s_weight + 0 * buf_elems + warp_id * warp_tile_elems;
        for (int row = 0; row < valid_rows; row++) {
            const __half* src = weight + (long long)(warp_base + row) * hidden_size;
            // cp.async 16 bytes at a time
            const int num_cps = tile_end / 8;
            __half* row_dst = dst + row * TILE_KK;
            for (int c = lane_id; c < num_cps; c += 32) {
                cp_async_16b(row_dst + c * 8, src + c * 8);
            }
            // Remainder
            for (int i = num_cps * 8 + lane_id; i < tile_end; i += 32) {
                row_dst[i] = src[i];
            }
        }
    }
    cp_async_commit_group();

    // MMA accumulators: 8 output values per warp, accumulated across K.
    // We do ceil(TILE_KK / 16) = 8 MMA ops per tile.
    // Each MMA produces 8 partial sums in row 0.
    float acc[TC_RPW];
    #pragma unroll
    for (int i = 0; i < TC_RPW; i++) acc[i] = 0.0f;

    for (int t = 0; t < num_tiles; t++) {
        const int buf_cur = t & 1;
        const int buf_nxt = 1 - buf_cur;
        const int tile_start = t * TILE_KK;
        const int tile_end = min((t + 1) * TILE_KK, hidden_size);
        const int tile_len = tile_end - tile_start;

        // Prefetch next tile
        if (t + 1 < num_tiles) {
            const int nxt_start = (t + 1) * TILE_KK;
            const int nxt_end = min((t + 2) * TILE_KK, hidden_size);
            const int nxt_len = nxt_end - nxt_start;
            __half* dst = s_weight + buf_nxt * buf_elems + warp_id * warp_tile_elems;
            for (int row = 0; row < valid_rows; row++) {
                const __half* src = weight + (long long)(warp_base + row) * hidden_size + nxt_start;
                __half* row_dst = dst + row * TILE_KK;
                const int num_cps = nxt_len / 8;
                for (int c = lane_id; c < num_cps; c += 32) {
                    cp_async_16b(row_dst + c * 8, src + c * 8);
                }
                for (int i = num_cps * 8 + lane_id; i < nxt_len; i += 32) {
                    row_dst[i] = src[i];
                }
            }
            cp_async_commit_group();
        }

        // Wait for current tile
        if (t + 1 < num_tiles) {
            cp_async_wait_group_1();
        } else {
            cp_async_wait_group_0();
        }
        __syncwarp();

        // Process current tile with MMA instructions
        // For each K-step of 16 within this tile:
        __half* w_tile = s_weight + buf_cur * buf_elems + warp_id * warp_tile_elems;

        const int k_steps = (tile_len + 15) / 16;
        for (int ks = 0; ks < k_steps; ks++) {
            const int k_off = ks * 16;
            const int k_valid = min(16, tile_len - k_off);

            // Load A fragment: broadcast input[tile_start + k_off .. +16]
            uint32_t a0, a1, a2, a3;
            if (k_valid == 16) {
                load_a_broadcast(a0, a1, a2, a3, s_input + tile_start + k_off, lane_id);
            } else {
                // Partial K tile: zero-pad
                __half tmp[16];
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    tmp[i] = (i < k_valid) ? s_input[tile_start + k_off + i] : __float2half(0.0f);
                }
                load_a_broadcast(a0, a1, a2, a3, tmp, lane_id);
            }

            // Pack 8 weight rows into B[16,8] col-major:
            //   B[k, n] = weight[warp_base + n, tile_start + k_off + k]
            //   D[0, n] = sum_k x[k] * weight[n, k] -- all 8 dot products in row 0
            //
            // Direct register load from strided weight tile (no staging):
            // Thread lane_id needs:
            //   n = lane_id / 4 (0..7), k0 = (lane_id % 4) * 2 (0,2,4,6)
            //   b0 = half2(w_tile[n * TILE_KK + k_off + k0], w_tile[n * TILE_KK + k_off + k0 + 1])
            //   b1 = half2(w_tile[n * TILE_KK + k_off + k0 + 8], w_tile[n * TILE_KK + k_off + k0 + 9])
            int n = lane_id / 4;
            int k0 = (lane_id % 4) * 2;
            uint32_t b0 = 0, b1 = 0;

            if (n < valid_rows && k_off + k0 + 1 < tile_len) {
                half2 v0 = *reinterpret_cast<const half2*>(&w_tile[n * TILE_KK + k_off + k0]);
                b0 = *reinterpret_cast<const uint32_t*>(&v0);
            }
            if (n < valid_rows && k_off + k0 + 9 < tile_len) {
                half2 v1 = *reinterpret_cast<const half2*>(&w_tile[n * TILE_KK + k_off + k0 + 8]);
                b1 = *reinterpret_cast<const uint32_t*>(&v1);
            }

            float d0, d1, d2, d3;
            mma_m16n8k16_f16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, 0.f, 0.f, 0.f, 0.f);

            // Row 0 of D held by lanes 0..3: d0=D[0,lane], d2=D[0,lane+4]
            if (lane_id < 4) {
                acc[lane_id] += d0;         // D[0, lane_id]
                acc[lane_id + 4] += d2;     // D[0, lane_id + 4]
            }
        }
    }

    // ---- Phase 3: Write output ----
    // Thread 0 has acc[0] (col 0) and acc[4] (col 4).
    // Thread 1 has acc[1] (col 1) and acc[5] (col 5).
    // Thread 2 has acc[2] (col 2) and acc[6] (col 6).
    // Thread 3 has acc[3] (col 3) and acc[7] (col 7).
    // Map: output[warp_base + lane_id] = acc[lane_id] for lanes 0..3
    //       output[warp_base + lane_id + 4] = acc[lane_id + 4] for lanes 0..3
    if (lane_id < 4) {
        if (lane_id < valid_rows)
            output[warp_base + lane_id] = __float2half(acc[lane_id]);
        if (lane_id + 4 < valid_rows)
            output[warp_base + lane_id + 4] = __float2half(acc[lane_id + 4]);
    }
}

// Variant with bias addition
extern "C"
__global__ void __launch_bounds__(TC_THREADS, 2)
tc_gemv_decode_bias_f16(
    __half* __restrict__ output,       // [out_dim]
    const __half* __restrict__ input,  // [hidden_size]
    const __half* __restrict__ weight, // [out_dim, hidden_size] row-major
    const __half* __restrict__ bias,   // [out_dim]
    int out_dim,
    int hidden_size
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int block_base = blockIdx.x * BLOCK_N;
    const int warp_base = block_base + warp_id * TC_RPW;

    extern __shared__ char smem_raw[];
    __half* s_input = (__half*)smem_raw;
    const int input_bytes = ((hidden_size * sizeof(__half)) + 127) & ~127;
    __half* s_weight = (__half*)(smem_raw + input_bytes);

    const int TILE_KK = 128;
    const int warp_tile_elems = TC_RPW * TILE_KK;
    const int buf_elems = TC_WARPS * warp_tile_elems;

    // Load input vector
    const int n_int4 = hidden_size / 8;
    const int4* input_int4 = (const int4*)input;
    int4* s_input_int4 = (int4*)s_input;
    for (int i = tid; i < n_int4; i += TC_THREADS) {
        s_input_int4[i] = input_int4[i];
    }
    for (int i = n_int4 * 8 + tid; i < hidden_size; i += TC_THREADS) {
        s_input[i] = input[i];
    }
    __syncthreads();

    if (warp_base >= out_dim) return;
    const int valid_rows = min(TC_RPW, out_dim - warp_base);

    const int num_tiles = (hidden_size + TILE_KK - 1) / TILE_KK;

    // Prefetch tile 0
    {
        const int tile_end = min(TILE_KK, hidden_size);
        __half* dst = s_weight + warp_id * warp_tile_elems;
        for (int row = 0; row < valid_rows; row++) {
            const __half* src = weight + (long long)(warp_base + row) * hidden_size;
            __half* row_dst = dst + row * TILE_KK;
            const int num_cps = tile_end / 8;
            for (int c = lane_id; c < num_cps; c += 32) {
                cp_async_16b(row_dst + c * 8, src + c * 8);
            }
            for (int i = num_cps * 8 + lane_id; i < tile_end; i += 32) {
                row_dst[i] = src[i];
            }
        }
    }
    cp_async_commit_group();

    float acc[TC_RPW];
    #pragma unroll
    for (int i = 0; i < TC_RPW; i++) acc[i] = 0.0f;

    for (int t = 0; t < num_tiles; t++) {
        const int buf_cur = t & 1;
        const int buf_nxt = 1 - buf_cur;
        const int tile_start = t * TILE_KK;
        const int tile_end = min((t + 1) * TILE_KK, hidden_size);
        const int tile_len = tile_end - tile_start;

        if (t + 1 < num_tiles) {
            const int nxt_start = (t + 1) * TILE_KK;
            const int nxt_end = min((t + 2) * TILE_KK, hidden_size);
            const int nxt_len = nxt_end - nxt_start;
            __half* dst = s_weight + buf_nxt * buf_elems + warp_id * warp_tile_elems;
            for (int row = 0; row < valid_rows; row++) {
                const __half* src = weight + (long long)(warp_base + row) * hidden_size + nxt_start;
                __half* row_dst = dst + row * TILE_KK;
                const int num_cps = nxt_len / 8;
                for (int c = lane_id; c < num_cps; c += 32) {
                    cp_async_16b(row_dst + c * 8, src + c * 8);
                }
                for (int i = num_cps * 8 + lane_id; i < nxt_len; i += 32) {
                    row_dst[i] = src[i];
                }
            }
            cp_async_commit_group();
        }

        if (t + 1 < num_tiles) {
            cp_async_wait_group_1();
        } else {
            cp_async_wait_group_0();
        }
        __syncwarp();

        __half* w_tile = s_weight + buf_cur * buf_elems + warp_id * warp_tile_elems;

        const int k_steps = (tile_len + 15) / 16;
        for (int ks = 0; ks < k_steps; ks++) {
            const int k_off = ks * 16;
            const int k_valid = min(16, tile_len - k_off);

            uint32_t a0, a1, a2, a3;
            if (k_valid == 16) {
                load_a_broadcast(a0, a1, a2, a3, s_input + tile_start + k_off, lane_id);
            } else {
                __half tmp[16];
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    tmp[i] = (i < k_valid) ? s_input[tile_start + k_off + i] : __float2half(0.0f);
                }
                load_a_broadcast(a0, a1, a2, a3, tmp, lane_id);
            }

            int n = lane_id / 4;
            int k0 = (lane_id % 4) * 2;
            uint32_t b0 = 0, b1 = 0;

            if (n < valid_rows && k_off + k0 + 1 < tile_len) {
                half2 v0 = *reinterpret_cast<const half2*>(&w_tile[n * TILE_KK + k_off + k0]);
                b0 = *reinterpret_cast<const uint32_t*>(&v0);
            }
            if (n < valid_rows && k_off + k0 + 9 < tile_len) {
                half2 v1 = *reinterpret_cast<const half2*>(&w_tile[n * TILE_KK + k_off + k0 + 8]);
                b1 = *reinterpret_cast<const uint32_t*>(&v1);
            }

            float d0, d1, d2, d3;
            mma_m16n8k16_f16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, 0.f, 0.f, 0.f, 0.f);

            if (lane_id < 4) {
                acc[lane_id] += d0;
                acc[lane_id + 4] += d2;     // D[0, lane_id + 4]
            }
        }
    }

    if (lane_id < 4) {
        if (lane_id < valid_rows)
            output[warp_base + lane_id] = __float2half(acc[lane_id] + __half2float(bias[warp_base + lane_id]));
        if (lane_id + 4 < valid_rows)
            output[warp_base + lane_id + 4] = __float2half(acc[lane_id + 4] + __half2float(bias[warp_base + lane_id + 4]));
    }
}

// Split-K variant: multiple blocks cooperate on K-dimension, atomicAdd partials.
// Launch with grid.y = num_k_splits, each block handles [k_start, k_end) slice.
extern "C"
__global__ void __launch_bounds__(TC_THREADS, 2)
tc_gemv_decode_splitk_f16(
    float* __restrict__ output_f32,    // [out_dim] f32 accumulator (zeroed before launch)
    const __half* __restrict__ input,  // [hidden_size]
    const __half* __restrict__ weight, // [out_dim, hidden_size] row-major
    int out_dim,
    int hidden_size,
    int k_start,
    int k_end
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int block_base = blockIdx.x * BLOCK_N;
    const int warp_base = block_base + warp_id * TC_RPW;

    const int k_len = k_end - k_start;

    extern __shared__ char smem_raw[];
    __half* s_input = (__half*)smem_raw;
    const int input_bytes = ((k_len * sizeof(__half)) + 127) & ~127;
    __half* s_weight = (__half*)(smem_raw + input_bytes);

    const int TILE_KK = 128;
    const int warp_tile_elems = TC_RPW * TILE_KK;
    const int buf_elems = TC_WARPS * warp_tile_elems;

    // Load input slice
    const int n_int4 = k_len / 8;
    const int4* input_int4 = (const int4*)(input + k_start);
    int4* s_input_int4 = (int4*)s_input;
    for (int i = tid; i < n_int4; i += TC_THREADS) {
        s_input_int4[i] = input_int4[i];
    }
    for (int i = n_int4 * 8 + tid; i < k_len; i += TC_THREADS) {
        s_input[i] = input[k_start + i];
    }
    __syncthreads();

    if (warp_base >= out_dim) return;
    const int valid_rows = min(TC_RPW, out_dim - warp_base);

    const int num_tiles = (k_len + TILE_KK - 1) / TILE_KK;

    // Prefetch tile 0
    {
        const int tile_end = min(TILE_KK, k_len);
        __half* dst = s_weight + warp_id * warp_tile_elems;
        for (int row = 0; row < valid_rows; row++) {
            const __half* src = weight + (long long)(warp_base + row) * hidden_size + k_start;
            __half* row_dst = dst + row * TILE_KK;
            const int num_cps = tile_end / 8;
            for (int c = lane_id; c < num_cps; c += 32) {
                cp_async_16b(row_dst + c * 8, src + c * 8);
            }
            for (int i = num_cps * 8 + lane_id; i < tile_end; i += 32) {
                row_dst[i] = src[i];
            }
        }
    }
    cp_async_commit_group();

    float acc[TC_RPW];
    #pragma unroll
    for (int i = 0; i < TC_RPW; i++) acc[i] = 0.0f;

    for (int t = 0; t < num_tiles; t++) {
        const int buf_cur = t & 1;
        const int buf_nxt = 1 - buf_cur;
        const int tile_start = t * TILE_KK;
        const int tile_end_local = min((t + 1) * TILE_KK, k_len);
        const int tile_len = tile_end_local - tile_start;

        if (t + 1 < num_tiles) {
            const int nxt_start = (t + 1) * TILE_KK;
            const int nxt_end = min((t + 2) * TILE_KK, k_len);
            const int nxt_len = nxt_end - nxt_start;
            __half* dst = s_weight + buf_nxt * buf_elems + warp_id * warp_tile_elems;
            for (int row = 0; row < valid_rows; row++) {
                const __half* src = weight + (long long)(warp_base + row) * hidden_size + k_start + nxt_start;
                __half* row_dst = dst + row * TILE_KK;
                const int num_cps = nxt_len / 8;
                for (int c = lane_id; c < num_cps; c += 32) {
                    cp_async_16b(row_dst + c * 8, src + c * 8);
                }
                for (int i = num_cps * 8 + lane_id; i < nxt_len; i += 32) {
                    row_dst[i] = src[i];
                }
            }
            cp_async_commit_group();
        }

        if (t + 1 < num_tiles) {
            cp_async_wait_group_1();
        } else {
            cp_async_wait_group_0();
        }
        __syncwarp();

        __half* w_tile = s_weight + buf_cur * buf_elems + warp_id * warp_tile_elems;

        const int k_steps = (tile_len + 15) / 16;
        for (int ks = 0; ks < k_steps; ks++) {
            const int k_off = ks * 16;
            const int k_valid = min(16, tile_len - k_off);

            uint32_t a0, a1, a2, a3;
            if (k_valid == 16) {
                load_a_broadcast(a0, a1, a2, a3, s_input + tile_start + k_off, lane_id);
            } else {
                __half tmp[16];
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    tmp[i] = (i < k_valid) ? s_input[tile_start + k_off + i] : __float2half(0.0f);
                }
                load_a_broadcast(a0, a1, a2, a3, tmp, lane_id);
            }

            int n = lane_id / 4;
            int k0_base = (lane_id % 4) * 2;
            uint32_t b0 = 0, b1 = 0;

            if (n < valid_rows && k_off + k0_base + 1 < tile_len) {
                half2 v0 = *reinterpret_cast<const half2*>(&w_tile[n * TILE_KK + k_off + k0_base]);
                b0 = *reinterpret_cast<const uint32_t*>(&v0);
            }
            if (n < valid_rows && k_off + k0_base + 9 < tile_len) {
                half2 v1 = *reinterpret_cast<const half2*>(&w_tile[n * TILE_KK + k_off + k0_base + 8]);
                b1 = *reinterpret_cast<const uint32_t*>(&v1);
            }

            float d0, d1, d2, d3;
            mma_m16n8k16_f16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, 0.f, 0.f, 0.f, 0.f);

            if (lane_id < 4) {
                acc[lane_id] += d0;
                acc[lane_id + 4] += d2;     // D[0, lane_id + 4]
            }
        }
    }

    // AtomicAdd partial results
    if (lane_id < 4) {
        if (lane_id < valid_rows)
            atomicAdd(&output_f32[warp_base + lane_id], acc[lane_id]);
        if (lane_id + 4 < valid_rows)
            atomicAdd(&output_f32[warp_base + lane_id + 4], acc[lane_id + 4]);
    }
}

// f32 -> f16 conversion for split-K output
extern "C"
__global__ void tc_splitk_f32_to_f16(
    __half* __restrict__ output,
    const float* __restrict__ input_f32,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __float2half(input_f32[idx]);
    }
}
