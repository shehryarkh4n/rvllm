// Standalone correctness harness for Hopper CUTLASS FFN paths.
//
// Validates the Hopper gate-aux FFN path against a CPU reference:
//   cutlass_hgemm(up) -> cutlass_gate_silu_mul -> cutlass_hgemm
//
// Compile on H100:
//   nvcc -O3 -std=c++17 test_cutlass_ffn.cu -ldl -lcudart -o test_cutlass_ffn
//
// Run:
//   ./test_cutlass_ffn ./sm_90/libcutlass_kernels.so

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define RT_CHECK(call) do {                                                    \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
        fprintf(stderr, "CUDA runtime error at %s:%d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));                  \
        std::exit(1);                                                          \
    }                                                                          \
} while (0)

using KernelFn = int (*)(void*, const void*, const void*, const void*, int, int, int, void*, size_t, void*);
using HgemmFn = int (*)(void*, const void*, const void*, int, int, int, void*, size_t, void*);
using WsFn = size_t (*)(int, int, int);

static uint32_t rng_state = 0x12345678u;

static uint32_t xorshift32() {
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return x;
}

static float rand_f32(float lo, float hi) {
    float t = (float)(xorshift32() & 0x00ffffffu) / (float)0x01000000u;
    return lo + (hi - lo) * t;
}

static __half rand_f16(float lo, float hi) {
    return __float2half(rand_f32(lo, hi));
}

template <typename T>
static void upload(T* dst, const std::vector<T>& src) {
    RT_CHECK(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
static void download(std::vector<T>& dst, const T* src) {
    RT_CHECK(cudaMemcpy(dst.data(), src, dst.size() * sizeof(T), cudaMemcpyDeviceToHost));
}

static float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

static void cpu_ref(
    const std::vector<__half>& x,
    const std::vector<__half>& w_gate,
    const std::vector<__half>& w_up,
    const std::vector<__half>& w_down,
    int m,
    int hidden,
    int intermediate,
    std::vector<float>& out)
{
    std::vector<float> gate(m * intermediate, 0.0f);
    std::vector<float> up(m * intermediate, 0.0f);
    std::vector<float> act(m * intermediate, 0.0f);

    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < intermediate; ++col) {
            float gv = 0.0f;
            float uv = 0.0f;
            for (int k = 0; k < hidden; ++k) {
                float xv = __half2float(x[row * hidden + k]);
                gv += xv * __half2float(w_gate[col * hidden + k]);
                uv += xv * __half2float(w_up[col * hidden + k]);
            }
            gate[row * intermediate + col] = gv;
            up[row * intermediate + col] = uv;
            act[row * intermediate + col] = silu(gv) * uv;
        }
    }

    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < hidden; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < intermediate; ++k) {
                sum += act[row * intermediate + k] * __half2float(w_down[col * intermediate + k]);
            }
            out[row * hidden + col] = sum;
        }
    }
}

static void compare_outputs(
    const char* label,
    const std::vector<__half>& got,
    const std::vector<float>& ref)
{
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    for (size_t i = 0; i < got.size(); ++i) {
        float diff = std::fabs(__half2float(got[i]) - ref[i]);
        max_abs = diff > max_abs ? diff : max_abs;
        mean_abs += diff;
    }
    mean_abs /= (float)got.size();
    printf("%s max_abs=%.6f mean_abs=%.6f\n", label, max_abs, mean_abs);
}

int main(int argc, char** argv) {
    const char* lib_path = argc > 1 ? argv[1] : "./sm_90/libcutlass_kernels.so";
    const int m = argc > 2 ? std::atoi(argv[2]) : 64;
    const int hidden = argc > 3 ? std::atoi(argv[3]) : 3584;
    const int intermediate = argc > 4 ? std::atoi(argv[4]) : 18944;

    void* lib = dlopen(lib_path, RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        fprintf(stderr, "dlopen(%s) failed: %s\n", lib_path, dlerror());
        return 1;
    }

    auto gateup_silu = (HgemmFn)dlsym(lib, "cutlass_gateup_silu");
    auto gateup_silu_ws = (WsFn)dlsym(lib, "cutlass_gateup_silu_workspace_size");
    auto gate_silu_mul = (KernelFn)dlsym(lib, "cutlass_gate_silu_mul");
    auto gate_silu_mul_ws = (WsFn)dlsym(lib, "cutlass_gate_silu_mul_workspace_size");
    auto hgemm = (HgemmFn)dlsym(lib, "cutlass_hgemm");
    auto hgemm_ws = (WsFn)dlsym(lib, "cutlass_hgemm_workspace_size");
    if (!gateup_silu || !gateup_silu_ws || !gate_silu_mul || !gate_silu_mul_ws || !hgemm || !hgemm_ws) {
        fprintf(stderr, "failed to resolve CUTLASS symbols\n");
        return 1;
    }

    std::vector<__half> x(m * hidden);
    std::vector<__half> w_gate(intermediate * hidden);
    std::vector<__half> w_up(intermediate * hidden);
    std::vector<__half> w_down(hidden * intermediate);
    std::vector<__half> w_gateup((size_t)2 * intermediate * hidden);

    for (auto& v : x) v = rand_f16(-0.25f, 0.25f);
    for (auto& v : w_gate) v = rand_f16(-0.05f, 0.05f);
    for (auto& v : w_up) v = rand_f16(-0.05f, 0.05f);
    for (auto& v : w_down) v = rand_f16(-0.05f, 0.05f);
    std::memcpy(w_gateup.data(), w_gate.data(), w_gate.size() * sizeof(__half));
    std::memcpy(w_gateup.data() + w_gate.size(), w_up.data(), w_up.size() * sizeof(__half));

    std::vector<float> ref((size_t)m * hidden, 0.0f);
    cpu_ref(x, w_gate, w_up, w_down, m, hidden, intermediate, ref);

    __half *d_x = nullptr, *d_w_gate = nullptr, *d_w_up = nullptr, *d_w_down = nullptr, *d_w_gateup = nullptr;
    __half *d_act_a = nullptr, *d_act_b = nullptr, *d_aux_up = nullptr, *d_out_a = nullptr, *d_out_b = nullptr;
    uint8_t *d_ws_a = nullptr, *d_ws_b = nullptr, *d_ws_hgemm = nullptr, *d_ws_up = nullptr;

    RT_CHECK(cudaMalloc(&d_x, x.size() * sizeof(__half)));
    RT_CHECK(cudaMalloc(&d_w_gate, w_gate.size() * sizeof(__half)));
    RT_CHECK(cudaMalloc(&d_w_up, w_up.size() * sizeof(__half)));
    RT_CHECK(cudaMalloc(&d_w_down, w_down.size() * sizeof(__half)));
    RT_CHECK(cudaMalloc(&d_w_gateup, w_gateup.size() * sizeof(__half)));
    RT_CHECK(cudaMalloc(&d_act_a, (size_t)m * intermediate * sizeof(__half)));
    RT_CHECK(cudaMalloc(&d_act_b, (size_t)m * intermediate * sizeof(__half)));
    RT_CHECK(cudaMalloc(&d_aux_up, (size_t)m * intermediate * sizeof(__half)));
    RT_CHECK(cudaMalloc(&d_out_a, (size_t)m * hidden * sizeof(__half)));
    RT_CHECK(cudaMalloc(&d_out_b, (size_t)m * hidden * sizeof(__half)));

    size_t ws_a = gateup_silu_ws(m, intermediate * 2, hidden);
    size_t ws_b = gate_silu_mul_ws(m, intermediate, hidden);
    size_t ws_up = hgemm_ws(m, intermediate, hidden);
    size_t ws_down = hgemm_ws(m, hidden, intermediate);
    RT_CHECK(cudaMalloc(&d_ws_a, ws_a ? ws_a : 1));
    RT_CHECK(cudaMalloc(&d_ws_b, ws_b ? ws_b : 1));
    RT_CHECK(cudaMalloc(&d_ws_up, ws_up ? ws_up : 1));
    RT_CHECK(cudaMalloc(&d_ws_hgemm, ws_down ? ws_down : 1));

    upload(d_x, x);
    upload(d_w_gate, w_gate);
    upload(d_w_up, w_up);
    upload(d_w_down, w_down);
    upload(d_w_gateup, w_gateup);

    cudaStream_t stream;
    RT_CHECK(cudaStreamCreate(&stream));

    fprintf(stderr, "stage=up_hgemm\n");
    fflush(stderr);
    if (hgemm(d_aux_up, d_x, d_w_up, m, intermediate, hidden, d_ws_up, ws_up, stream) != 0) {
        fprintf(stderr, "up hgemm failed\n");
        return 1;
    }
    fprintf(stderr, "stage=gateup_silu\n");
    fflush(stderr);
    if (gateup_silu(d_act_a, d_x, d_w_gateup, m, intermediate * 2, hidden, d_ws_a, ws_a, stream) != 0) {
        fprintf(stderr, "cutlass_gateup_silu failed\n");
        return 1;
    }
    fprintf(stderr, "stage=down_after_gateup_silu\n");
    fflush(stderr);
    if (hgemm(d_out_a, d_act_a, d_w_down, m, hidden, intermediate, d_ws_hgemm, ws_down, stream) != 0) {
        fprintf(stderr, "down hgemm after gateup_silu failed\n");
        return 1;
    }
    fprintf(stderr, "stage=gate_silu_mul\n");
    fflush(stderr);
    if (gate_silu_mul(d_act_b, d_x, d_w_gate, d_aux_up, m, intermediate, hidden, d_ws_b, ws_b, stream) != 0) {
        fprintf(stderr, "cutlass_gate_silu_mul failed\n");
        return 1;
    }
    fprintf(stderr, "stage=down_after_gate_silu_mul\n");
    fflush(stderr);
    if (hgemm(d_out_b, d_act_b, d_w_down, m, hidden, intermediate, d_ws_hgemm, ws_down, stream) != 0) {
        fprintf(stderr, "down hgemm after gate_silu_mul failed\n");
        return 1;
    }

    fprintf(stderr, "stage=sync\n");
    fflush(stderr);
    RT_CHECK(cudaStreamSynchronize(stream));

    std::vector<__half> out_a((size_t)m * hidden);
    std::vector<__half> out_b((size_t)m * hidden);
    std::vector<__half> act_a((size_t)m * intermediate);
    std::vector<__half> act_b((size_t)m * intermediate);
    download(out_a, d_out_a);
    download(out_b, d_out_b);
    download(act_a, d_act_a);
    download(act_b, d_act_b);

    compare_outputs("gateup_silu -> down vs cpu", out_a, ref);
    compare_outputs("gate_silu_mul -> down vs cpu", out_b, ref);
    printf("activated tensor elems=%zu\n", act_a.size());

    RT_CHECK(cudaStreamDestroy(stream));
    cudaFree(d_x);
    cudaFree(d_w_gate);
    cudaFree(d_w_up);
    cudaFree(d_w_down);
    cudaFree(d_w_gateup);
    cudaFree(d_act_a);
    cudaFree(d_act_b);
    cudaFree(d_aux_up);
    cudaFree(d_out_a);
    cudaFree(d_out_b);
    cudaFree(d_ws_a);
    cudaFree(d_ws_b);
    cudaFree(d_ws_up);
    cudaFree(d_ws_hgemm);
    dlclose(lib);
    return 0;
}
