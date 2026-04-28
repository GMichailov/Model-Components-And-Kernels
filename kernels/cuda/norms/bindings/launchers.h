#pragma once

#include <torch/extension.h>
#include "../src/rmsnorm.cuh"

using namespace norms::launchers;

// Simple wrapper launchers for testing
inline at::Tensor rmsnorm_online_bf16_bf16_f32_t32(const at::Tensor& x, const at::Tensor& gamma, float epsilon, int dim_x) {
    return online::cuda_rms_norm_forward_launcher_online<at::BFloat16, at::BFloat16, float, 32>(x, gamma, epsilon, dim_x);
}
