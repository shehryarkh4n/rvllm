// CUTLASS 3.x SM90 GEMM + per-column bias epilogue for QKV projection.
//
// D[M,N] = A[M,K] @ B[K,N]^T + bias[N]
//
// bias is broadcast across the M dimension (per-column bias).
// Shapes for Qwen2.5-7B: M=128, N=4608, K=3584.
//
// Build: nvcc -std=c++17 -arch=sm_90a --expt-relaxed-constexpr \
//        -I/root/cutlass/include -I/root/cutlass/tools/util/include \
//        -O3 -o libcutlass_qkv_bias.so --shared -Xcompiler -fPIC cutlass_qkv_bias.cu

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/operations.hpp>
#include <cute/tensor.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <cuda_fp16.h>

using namespace cute;

// ============================================================================
// Type aliases
// ============================================================================

using ElementA     = cutlass::half_t;
using ElementB     = cutlass::half_t;
using ElementC     = cutlass::half_t;   // bias type
using ElementD     = cutlass::half_t;   // output type
using ElementAccum = float;
using ElementCompute = float;

// A is [M, K] row-major
// B is [N, K] row-major, accessed as column-major (transposed)
// D is [M, N] row-major
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;

// Tile 128x128x64 for Hopper WGMMA
using TileShape    = Shape<_128, _128, _64>;
using ClusterShape = Shape<_1, _1, _1>;

// ============================================================================
// Epilogue: D = alpha * accum + beta * C + bias[N]
// Using built-in LinCombPerColBias (beta=0, alpha=1.0)
// ============================================================================

using FusionOp = cutlass::epilogue::fusion::LinCombPerColBias<
    ElementD,        // output
    ElementCompute,  // compute
    ElementC         // bias element type
>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccum, ElementCompute,
    ElementD, LayoutD, 8,  // C (unused, beta=0)
    ElementD, LayoutD, 8,  // D
    cutlass::epilogue::TmaWarpSpecialized,
    FusionOp
>::CollectiveOp;

// ============================================================================
// Collective mainloop
// ============================================================================

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, 8,
    ElementB, LayoutB, 8,
    ElementAccum,
    TileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

// ============================================================================
// GEMM kernel + adapter
// ============================================================================

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// ============================================================================
// C interface
// ============================================================================

extern "C" {

int cutlass_qkv_bias_gemm(
    void* output,          // [M, N] half
    const void* input,     // [M, K] half
    const void* weight,    // [N, K] half (row-major, transposed in GEMM)
    const void* bias,      // [N] half
    int M, int N, int K,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream
) {
    auto prob_shape = cute::make_shape(M, N, K, 1);

    auto stride_A = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_D = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideD{}, {M, N, 1});

    typename CollectiveEpilogue::FusionCallbacks::Arguments fusion_args;
    fusion_args.alpha = ElementCompute(1.0f);
    fusion_args.beta = ElementCompute(0.0f);
    fusion_args.bias_ptr = reinterpret_cast<const ElementC*>(bias);

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        prob_shape,
        {   // mainloop args
            reinterpret_cast<const ElementA*>(input), stride_A,
            reinterpret_cast<const ElementB*>(weight), stride_B,
        },
        {   // epilogue args
            fusion_args,
            nullptr, stride_D,  // C ptr and stride (unused, beta=0)
            reinterpret_cast<ElementD*>(output), stride_D   // D
        }
    };

    Gemm gemm_op;
    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) return -1;

    status = gemm_op.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) return -2;

    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) return -3;

    return 0;
}

size_t cutlass_qkv_bias_workspace_size(int M, int N, int K) {
    auto prob_shape = cute::make_shape(M, N, K, 1);

    auto stride_A = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_D = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideD{}, {M, N, 1});

    typename CollectiveEpilogue::FusionCallbacks::Arguments fusion_args;
    fusion_args.alpha = ElementCompute(1.0f);
    fusion_args.beta = ElementCompute(0.0f);
    fusion_args.bias_ptr = nullptr;

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        prob_shape,
        {nullptr, stride_A, nullptr, stride_B},
        {fusion_args, nullptr, stride_D, nullptr, stride_D}
    };

    Gemm gemm_op;
    return gemm_op.get_workspace_size(args);
}

} // extern "C"
