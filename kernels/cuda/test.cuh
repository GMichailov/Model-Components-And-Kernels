#pragma once

#include "norms/rmsnorm.cuh"
#include <torch/torch.h>
#include <torch/nn/functional/normalization.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <iostream>
#include <string.h>

// Test Correctness functions
// Test for a handful of datatypes and combinations.

namespace test {

namespace utils {

    bool allclose_cuda(const torch::Tensor& a, const torch::Tensor& b, double rtol, double atol) { return torch::allclose(a, b, rtol, atol); }

    namespace rms_norm {

        inline void log_failure_debug_rms(std::string& test_name, const torch::Tensor& y_me, const torch::Tensor& y_ref, int rows, int dim_x, torch::ScalarType act_dtype, torch::ScalarType wgt_dtype, float eps) {
            auto y_me_fp32 = y_me.to(torch::kFloat32);
            auto y_ref_fp32 = y_ref.to(torch::kFloat32);
            auto diff = (y_me_fp32 - y_ref_fp32).abs();
            std::cerr << "Test failed:" << test_name << "\n";
            std::cerr << "rows=" << rows << " dim_x=" << dim_x << " eps=" << eps
                    << " act_dtype=" << act_dtype << " weight_dtype=" << wgt_dtype
                    << " max_abs_diff=" << diff.max().item<float>() << "\n";
        }

        inline torch::Tensor generate_fake_tensor(int64_t seq_len, int64_t dim_x, torch::ScalarType dtype) {
            return torch::randn({seq_len, dim_x}, torch::TensorOptions().dtype(dtype).device(torch::kCUDA));
        }

        inline torch::Tensor manual_rmsnorm_forward(const torch::Tensor& x, const torch::Tensor& gamma, float epsilon) {
            auto x_fp32 = x.to(torch::kFloat);
            auto ms = x_fp32.pow(2).mean(-1, true);
            auto y = x_fp32 * torch::rsqrt(ms + epsilon);
            if (gamma.defined()) {
                y = y * gamma.to(torch::kFloat);
            }
            return y.to(x.scalar_type());
        }

        template<typename ACTIVATION_DTYPE, typename WEIGHT_DTYPE, typename VECTORIZED_LOAD_TYPE, typename CALCULATION_TYPE, int DIM_X>
        bool run_rms_norm_forward_divisible_case(const std::string& case_name, int rows, float epsilon, int num_threads, torch::ScalarType act_dtype, torch::ScalarType wgt_dtype, double rtol, double atol) {
            try {
                auto x = generate_fake_tensor(rows, DIM_X, act_dtype).contiguous();
                auto gamma = generate_fake_tensor(1, DIM_X, WEIGHT_DTYPE).contiguous();
                auto y_me = norm::launchers::offline<ACTIVATION_DTYPE, WEIGHT_DTYPE, VECTORIZED_LOAD_TYPE, CALCULATION_TYPE, DIM_X>(x, gamma, epsilon, num_threads);
                cudaError_t err = cudaDeviceSynchronize();
                if (err != cudaSuccess) {
                    std::cerr << "Kernel execution failed: " << case_name << " -> " << cudaGetErrorString(err) << "\n";
                    return false;
                }
                auto y_ref = manual_rmsnorm_forward(x, gamma, epsilon);
                bool pass = allclose_cuda(y_me, y_ref, rtol, atol);
                if (!pass) log_failure_debug_rms(case_name, y_me, y_ref, rows, DIM_X, act_dtype, wgt_dtype, epsilon);
                return pass;
            } catch (const c10::Error& e) {
                std::cerr << "Torch error in " << case_name << ": " << e.what() << "\n";
                return false;
            } catch (const std::exception& e) {
                std::cerr << "Std exception in " << case_name << ": " << e.what() << "\n";
                return false;
            }
        }

    }

}

inline bool test_rms_norm_forward_divisible() {
    torch::NoGradGuard no_grad;
    torch::manual_seed(0);
    std::string test_name{"RMSNorm forward (divisible)"};
    bool all_pass{true};
    // Tests for:
    // template<typename ACTIVATION_DTYPE, typename WEIGHT_DTYPE,
    //          typename VECTORIZED_LOAD_TYPE, typename CALCULATION_DTYPE, int DIM_X>
    // torch::Tensor cuda_rmsnorm_divisble_forward_launcher_offline(
    //     const torch::Tensor& x,
    //     const torch::Tensor& gamma,
    //     float epsilon,
    //     int num_threads);
    //
    // This test suite covers the "divisible" RMSNorm forward launcher variant, which assumes
    // each row can be cleanly consumed by vectorized loads with no tail handling required.
    //
    // Compile-time assumptions enforced by the launcher/kernel:
    // 1. (DIM_X * sizeof(ACTIVATION_DTYPE)) % sizeof(VECTORIZED_LOAD_TYPE) == 0
    //    -> A full row must be evenly divisible by the vectorized load width in bytes.
    //
    // 2. sizeof(VECTORIZED_LOAD_TYPE) % sizeof(ACTIVATION_DTYPE) == 0
    //    -> The vector type must hold an integer number of activation elements.
    //
    // 3. sizeof(VECTORIZED_LOAD_TYPE) >= sizeof(ACTIVATION_DTYPE)
    //    -> The vectorized load type must be at least as large as a single activation element.
    //
    // 4. sizeof(CALCULATION_DTYPE) >= sizeof(ACTIVATION_DTYPE)
    //    -> The accumulation / calculation type must be at least as wide as the activation type.
    //
    // Test intent:
    // - Validate numerical correctness of the divisible vectorized forward RMSNorm path.
    // - Sweep epsilon values to check numerical stability sensitivity.
    // - Sweep thread counts to verify launch configuration correctness.
    // - Sweep DIM_X values that still satisfy clean vectorization constraints.
    //
    // Params are ordered as:
    // ACTIVATION_DTYPE, WEIGHT_DTYPE, VECTORIZED_LOAD_TYPE, CALCULATION_DTYPE, DIM_X, EPSILON, NUM_THREADS
    //
    // Test 1: fp32, fp32, fp32, fp32, 1024, 1e-6, 256
    //   - Baseline correctness check with standard epsilon and moderate thread count.
    //
    // Test 2: fp32, fp32, fp32, fp32, 1024, 1e-8, 256
    //   - Same shape and launch config, but tighter epsilon for numerical stability coverage.
    //
    // Test 3: fp32, fp32, fp32, fp32, 1024, 1e-8, 512
    //   - Same data shape as Test 2, but higher thread count to validate launch behavior.
    //
    // Test 4: fp32, fp32, fp32, fp32, 512, 1e-8, 1024
    //   - Smaller DIM_X with aggressive thread count to stress configuration assumptions.
    // Test 5: fp16, fp16, half2, fp32, 1024, 1e-6, 256
    //   - Standard half-precision path with packed half2 vector loads and fp32 accumulation.
    //
    // Test 6: fp16, fp16, half2, fp32, 1024, 1e-8, 256
    //   - Same as Test 5, but tighter epsilon for numerical stability coverage.
    //
    // Test 7: fp16, fp16, float, fp32, 1024, 1e-6, 256
    //   - Half activations loaded through a 4-byte vector type; still valid since
    //     sizeof(float) is a multiple of sizeof(fp16) and row size remains divisible.
    //
    // Test 8: fp16, fp16, float2, fp32, 1024, 1e-6, 256
    //   - Wider vectorized load for fp16 data; useful for checking larger packed loads.
    //
    // Test 9: bf16, bf16, __nv_bfloat162, fp32, 1024, 1e-6, 256
    //   - Standard bf16 path with packed bf16x2 loads and fp32 accumulation.
    //
    // Test 10: bf16, bf16, __nv_bfloat162, fp32, 1024, 1e-8, 256
    //   - Same as Test 9, but tighter epsilon for numerical stability coverage.
    //
    // Test 11: bf16, bf16, float, fp32, 1024, 1e-6, 256
    //   - bf16 activations using a 4-byte vectorized load type; valid if implementation
    //     treats the vectorized type as a load container rather than a semantic float value.
    //
    // Test 12: bf16, bf16, float2, fp32, 1024, 1e-6, 256
    //   - Wider vectorized load for bf16 path to exercise larger packed memory reads.
    //
    // Test 13: fp32, fp16, float, fp32, 1024, 1e-6, 256
    //   - Mixed activation / weight precision with fp32 activations and fp16 weights.
    //     Safe under the stated assumption that activation type size is >= weight type size.
    //
    // Test 14: fp32, bf16, float, fp32, 1024, 1e-6, 256
    //   - Mixed activation / weight precision with fp32 activations and bf16 weights.
    //     Useful for validating lower-precision scale storage with full-precision activations.
    //
    // Test 15: fp16, fp16, half2, fp16, 1024, 1e-6, 256
    //   - Pure half-precision path including accumulation type, if intentionally supported.
    //     Less numerically stable than fp32 accumulation, but still satisfies the static asserts.
    //
    // Test 16: bf16, bf16, __nv_bfloat162, bf16, 1024, 1e-6, 256
    //   - Pure bf16 path including accumulation type, if intentionally supported.
    //     Primarily useful as a stress / behavior test rather than a preferred accuracy path.
    // Test 17: fp32, fp32, float4, fp32, 1024, 1e-6, 256
    //   - Wider 16-byte vectorized load for fp32 data; useful for validating larger packed loads
    //     when each row is still evenly divisible by the vector width.
    //
    // Test 18: fp32, bf16, float4, fp32, 1024, 1e-6, 256
    //   - fp32 activations with lower-precision bf16 weights and 16-byte vectorized loads.
    //     Safe because the activation row size remains divisible by float4 loads and
    //     activation type size is >= weight type size.
    //
    // Test 19: bf16, bf16, __nv_bfloat162, bf16, 1024, 1e-6, 256
    //   - Pure bf16 path including bf16 activations, bf16 weights, packed bf16x2 loads,
    //     and bf16 accumulation, if intentionally supported by the implementation.
    //
    // Test 20: bf16, bf16, __nv_bfloat162, fp32, 1024, 1e-6, 256
    //   - Preferred bf16 execution path with packed bf16x2 loads and fp32 accumulation.
    constexpr int rows = 2048;
    #define RUN_DIV_CASE(ID, A, W, V, C, DIM, EPS, THREADS, ACT_SCALAR, WGT_SCALAR, RTOL, ATOL)           \
    do {                                                                                                \
        if (!utils::rms_norm::run_rms_norm_forward_divisible_case<A, W, V, C, DIM>(                                            \
                test_name + " / case " #ID,                                                             \
                rows,                                                                                   \
                EPS,                                                                                    \
                THREADS,                                                                                \
                ACT_SCALAR,                                                                             \
                WGT_SCALAR,                                                                             \
                RTOL,                                                                                   \
                ATOL)) {                                                                                \
            all_pass = false;                                                                           \
        }                                                                                               \
    } while (0)

    RUN_DIV_CASE(1,  float,        float,        float,          float,        1024, 1e-6f, 256, torch::kFloat32,  torch::kFloat32,  1e-4, 1e-5);
    RUN_DIV_CASE(2,  float,        float,        float,          float,        1024, 1e-8f, 256, torch::kFloat32,  torch::kFloat32,  1e-4, 1e-5);
    RUN_DIV_CASE(3,  float,        float,        float,          float,        1024, 1e-8f, 512, torch::kFloat32,  torch::kFloat32,  1e-4, 1e-5);
    RUN_DIV_CASE(4,  float,        float,        float,          float,         512, 1e-8f,1024, torch::kFloat32,  torch::kFloat32,  1e-4, 1e-5);

    RUN_DIV_CASE(5,  at::Half,     at::Half,     half2,          float,        1024, 1e-6f, 256, torch::kFloat16,  torch::kFloat16,  5e-3, 5e-3);
    RUN_DIV_CASE(6,  at::Half,     at::Half,     half2,          float,        1024, 1e-8f, 256, torch::kFloat16,  torch::kFloat16,  5e-3, 5e-3);
    RUN_DIV_CASE(7,  at::Half,     at::Half,     float,          float,        1024, 1e-6f, 256, torch::kFloat16,  torch::kFloat16,  5e-3, 5e-3);
    RUN_DIV_CASE(8,  at::Half,     at::Half,     float2,         float,        1024, 1e-6f, 256, torch::kFloat16,  torch::kFloat16,  5e-3, 5e-3);

    RUN_DIV_CASE(9,  at::BFloat16, at::BFloat16, __nv_bfloat162, float,        1024, 1e-6f, 256, torch::kBFloat16, torch::kBFloat16, 2e-2, 2e-2);
    RUN_DIV_CASE(10, at::BFloat16, at::BFloat16, __nv_bfloat162, float,        1024, 1e-8f, 256, torch::kBFloat16, torch::kBFloat16, 2e-2, 2e-2);
    RUN_DIV_CASE(11, at::BFloat16, at::BFloat16, float,          float,        1024, 1e-6f, 256, torch::kBFloat16, torch::kBFloat16, 2e-2, 2e-2);
    RUN_DIV_CASE(12, at::BFloat16, at::BFloat16, float2,         float,        1024, 1e-6f, 256, torch::kBFloat16, torch::kBFloat16, 2e-2, 2e-2);

    RUN_DIV_CASE(13, float,        at::Half,     float,          float,        1024, 1e-6f, 256, torch::kFloat32,  torch::kFloat16,  1e-4, 5e-5);
    RUN_DIV_CASE(14, float,        at::BFloat16, float,          float,        1024, 1e-6f, 256, torch::kFloat32,  torch::kBFloat16, 1e-4, 5e-5);

    RUN_DIV_CASE(15, at::Half,     at::Half,     half2,          at::Half,     1024, 1e-6f, 256, torch::kFloat16,  torch::kFloat16,  5e-2, 5e-2);
    RUN_DIV_CASE(16, at::BFloat16, at::BFloat16, __nv_bfloat162, at::BFloat16, 1024, 1e-6f, 256, torch::kBFloat16, torch::kBFloat16, 7e-2, 7e-2);

    RUN_DIV_CASE(17, float,        float,        float4,         float,        1024, 1e-6f, 256, torch::kFloat32,  torch::kFloat32,  1e-4, 1e-5);
    RUN_DIV_CASE(18, float,        at::BFloat16, float4,         float,        1024, 1e-6f, 256, torch::kFloat32,  torch::kBFloat16, 1e-4, 5e-5);
    RUN_DIV_CASE(19, at::BFloat16, at::BFloat16, __nv_bfloat162, at::BFloat16, 1024, 1e-6f, 256, torch::kBFloat16, torch::kBFloat16, 7e-2, 7e-2);
    RUN_DIV_CASE(20, at::BFloat16, at::BFloat16, __nv_bfloat162, float,        1024, 1e-6f, 256, torch::kBFloat16, torch::kBFloat16, 2e-2, 2e-2);

    #undef RUN_DIV_CASE
        return all_pass;
    }

inline bool test_rms_norm_forward_indivisible() {}
    
}