#pragma once

#include <torch/extension.h>
#include "../src/rmsnorm.cuh"

using namespace norms::launchers::offline;

// ============================================================================
// FP32/FP32/FP32 (ACT=float, WGT=float, CALC=float)
// ============================================================================

at::Tensor rmsnorm_forward_fp32_fp32_fp32_v1(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (x.size(1)) {
        case 128:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 1, 128, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 1, 128, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 1, 128, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 256:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 1, 256, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 1, 256, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 1, 256, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 512:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 1, 512, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 1, 512, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 1, 512, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        default:
            TORCH_CHECK(false, "Unsupported hidden dim");
    }
}

at::Tensor rmsnorm_forward_fp32_fp32_fp32_v2(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (x.size(1)) {
        case 128:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 2, 128, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 2, 128, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 2, 128, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 256:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 2, 256, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 2, 256, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 2, 256, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 512:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 2, 512, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 2, 512, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 2, 512, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        default:
            TORCH_CHECK(false, "Unsupported hidden dim");
    }
}

at::Tensor rmsnorm_forward_fp32_fp32_fp32_v4(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (x.size(1)) {
        case 128:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 4, 128, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 4, 128, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 4, 128, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 256:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 4, 256, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 4, 256, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 4, 256, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 512:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 4, 512, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 4, 512, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 4, 512, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        default:
            TORCH_CHECK(false, "Unsupported hidden dim");
    }
}

// ============================================================================
// FP16/FP16/FP32 (ACT=at::Half, WGT=at::Half, CALC=float)
// ============================================================================

at::Tensor rmsnorm_forward_fp16_fp16_fp32_v1(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (x.size(1)) {
        case 128:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 1, 128, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 1, 128, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 1, 128, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 256:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 1, 256, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 1, 256, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 1, 256, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 512:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 1, 512, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 1, 512, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 1, 512, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        default:
            TORCH_CHECK(false, "Unsupported hidden dim");
    }
}

at::Tensor rmsnorm_forward_fp16_fp16_fp32_v2(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (x.size(1)) {
        case 128:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 2, 128, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 2, 128, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 2, 128, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 256:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 2, 256, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 2, 256, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 2, 256, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 512:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 2, 512, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 2, 512, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 2, 512, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        default:
            TORCH_CHECK(false, "Unsupported hidden dim");
    }
}

at::Tensor rmsnorm_forward_fp16_fp16_fp32_v4(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (x.size(1)) {
        case 128:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 4, 128, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 4, 128, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 4, 128, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 256:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 4, 256, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 4, 256, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 4, 256, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 512:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 4, 512, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 4, 512, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 4, 512, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        default:
            TORCH_CHECK(false, "Unsupported hidden dim");
    }
}

at::Tensor rmsnorm_forward_fp16_fp16_fp32_v8(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (x.size(1)) {
        case 128:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 8, 128, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 8, 128, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 8, 128, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 256:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 8, 256, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 8, 256, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 8, 256, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 512:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 8, 512, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 8, 512, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 8, 512, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        default:
            TORCH_CHECK(false, "Unsupported hidden dim");
    }
}

// ============================================================================
// FP16/FP32/FP32 (ACT=at::Half, WGT=float, CALC=float)
// ============================================================================

at::Tensor rmsnorm_forward_fp16_fp32_fp32_v1(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (x.size(1)) {
        case 128:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 1, 128, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 1, 128, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 1, 128, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 256:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 1, 256, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 1, 256, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 1, 256, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 512:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 1, 512, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 1, 512, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 1, 512, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        default:
            TORCH_CHECK(false, "Unsupported hidden dim");
    }
}

at::Tensor rmsnorm_forward_fp16_fp32_fp32_v2(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (x.size(1)) {
        case 128:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 2, 128, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 2, 128, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 2, 128, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 256:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 2, 256, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 2, 256, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 2, 256, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 512:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 2, 512, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 2, 512, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 2, 512, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        default:
            TORCH_CHECK(false, "Unsupported hidden dim");
    }
}

at::Tensor rmsnorm_forward_fp16_fp32_fp32_v4(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (x.size(1)) {
        case 128:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 4, 128, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 4, 128, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 4, 128, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 256:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 4, 256, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 4, 256, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 4, 256, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 512:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 4, 512, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 4, 512, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 4, 512, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        default:
            TORCH_CHECK(false, "Unsupported hidden dim");
    }
}

at::Tensor rmsnorm_forward_fp16_fp32_fp32_v8(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (x.size(1)) {
        case 128:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 8, 128, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 8, 128, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 8, 128, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 256:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 8, 256, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 8, 256, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 8, 256, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 512:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 8, 512, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 8, 512, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 8, 512, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        default:
            TORCH_CHECK(false, "Unsupported hidden dim");
    }
}

// ============================================================================
// BF16/BF16/FP32 (ACT=at::BFloat16, WGT=at::BFloat16, CALC=float)
// ============================================================================

at::Tensor rmsnorm_forward_bf16_bf16_fp32_v1(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (x.size(1)) {
        case 128:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 128, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 128, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 128, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 256:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 256, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 256, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 256, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 512:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 512, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 512, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 512, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        default:
            TORCH_CHECK(false, "Unsupported hidden dim");
    }
}

at::Tensor rmsnorm_forward_bf16_bf16_fp32_v2(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (x.size(1)) {
        case 128:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 128, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 128, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 128, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 256:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 256, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 256, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 256, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 512:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 512, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 512, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 512, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        default:
            TORCH_CHECK(false, "Unsupported hidden dim");
    }
}

at::Tensor rmsnorm_forward_bf16_bf16_fp32_v4(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (x.size(1)) {
        case 128:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 128, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 128, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 128, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 256:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 256, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 256, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 256, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 512:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 512, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 512, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 512, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        default:
            TORCH_CHECK(false, "Unsupported hidden dim");
    }
}

at::Tensor rmsnorm_forward_bf16_bf16_fp32_v8(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (x.size(1)) {
        case 128:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 128, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 128, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 128, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 256:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 256, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 256, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 256, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 512:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 512, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 512, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 512, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        default:
            TORCH_CHECK(false, "Unsupported hidden dim");
    }
}

// ============================================================================
// BF16/FP32/FP32 (ACT=at::BFloat16, WGT=float, CALC=float)
// ============================================================================

at::Tensor rmsnorm_forward_bf16_fp32_fp32_v1(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (x.size(1)) {
        case 128:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 1, 128, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 1, 128, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 1, 128, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 256:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 1, 256, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 1, 256, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 1, 256, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 512:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 1, 512, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 1, 512, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 1, 512, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        default:
            TORCH_CHECK(false, "Unsupported hidden dim");
    }
}

at::Tensor rmsnorm_forward_bf16_fp32_fp32_v2(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (x.size(1)) {
        case 128:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 2, 128, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 2, 128, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 2, 128, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 256:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 2, 256, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 2, 256, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 2, 256, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 512:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 2, 512, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 2, 512, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 2, 512, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        default:
            TORCH_CHECK(false, "Unsupported hidden dim");
    }
}

at::Tensor rmsnorm_forward_bf16_fp32_fp32_v4(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (x.size(1)) {
        case 128:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 4, 128, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 4, 128, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 4, 128, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 256:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 4, 256, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 4, 256, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 4, 256, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 512:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 4, 512, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 4, 512, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 4, 512, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        default:
            TORCH_CHECK(false, "Unsupported hidden dim");
    }
}

at::Tensor rmsnorm_forward_bf16_fp32_fp32_v8(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (x.size(1)) {
        case 128:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 8, 128, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 8, 128, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 8, 128, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 256:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 8, 256, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 8, 256, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 8, 256, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 512:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 8, 512, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 8, 512, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 8, 512, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        default:
            TORCH_CHECK(false, "Unsupported hidden dim");
    }
}

// ============================================================================
// Indivisible forward launchers
// ============================================================================

// FP32/FP32/FP32 (ACT=float, WGT=float, CALC=float)

at::Tensor rmsnorm_forward_fp32_fp32_fp32_indivisible_v1(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (x.size(1)) {
        case 128:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 1, 128, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 1, 128, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 1, 128, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 256:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 1, 256, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 1, 256, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 1, 256, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 512:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 1, 512, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 1, 512, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 1, 512, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        default:
            TORCH_CHECK(false, "Unsupported hidden dim");
    }
}

at::Tensor rmsnorm_forward_fp32_fp32_fp32_indivisible_v2(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (x.size(1)) {
        case 128:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 2, 128, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 2, 128, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 2, 128, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 256:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 2, 256, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 2, 256, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 2, 256, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 512:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 2, 512, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 2, 512, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 2, 512, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        default:
            TORCH_CHECK(false, "Unsupported hidden dim");
    }
}

at::Tensor rmsnorm_forward_fp32_fp32_fp32_indivisible_v4(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (x.size(1)) {
        case 128:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 4, 128, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 4, 128, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 4, 128, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 256:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 4, 256, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 4, 256, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 4, 256, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 512:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 4, 512, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 4, 512, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 4, 512, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        default:
            TORCH_CHECK(false, "Unsupported hidden dim");
    }
}

// FP16/FP16/FP32 (ACT=at::Half, WGT=at::Half, CALC=float)

at::Tensor rmsnorm_forward_fp16_fp16_fp32_indivisible_v1(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (x.size(1)) {
        case 128:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 1, 128, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 1, 128, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 1, 128, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 256:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 1, 256, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 1, 256, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 1, 256, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 512:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 1, 512, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 1, 512, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 1, 512, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        default:
            TORCH_CHECK(false, "Unsupported hidden dim");
    }
}

at::Tensor rmsnorm_forward_fp16_fp16_fp32_indivisible_v2(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (x.size(1)) {
        case 128:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 2, 128, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 2, 128, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 2, 128, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 256:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 2, 256, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 2, 256, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 2, 256, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 512:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 2, 512, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 2, 512, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 2, 512, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        default:
            TORCH_CHECK(false, "Unsupported hidden dim");
    }
}

at::Tensor rmsnorm_forward_fp16_fp16_fp32_indivisible_v4(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (x.size(1)) {
        case 128:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 4, 128, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 4, 128, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 4, 128, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 256:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 4, 256, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 4, 256, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 4, 256, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 512:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 4, 512, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 4, 512, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 4, 512, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        default:
            TORCH_CHECK(false, "Unsupported hidden dim");
    }
}

at::Tensor rmsnorm_forward_fp16_fp16_fp32_indivisible_v8(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (x.size(1)) {
        case 128:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 8, 128, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 8, 128, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 8, 128, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 256:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 8, 256, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 8, 256, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 8, 256, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 512:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 8, 512, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 8, 512, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 8, 512, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        default:
            TORCH_CHECK(false, "Unsupported hidden dim");
    }
}

// FP16/FP32/FP32 (ACT=at::Half, WGT=float, CALC=float)

at::Tensor rmsnorm_forward_fp16_fp32_fp32_indivisible_v1(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (x.size(1)) {
        case 128:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 1, 128, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 1, 128, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 1, 128, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 256:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 1, 256, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 1, 256, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 1, 256, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 512:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 1, 512, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 1, 512, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 1, 512, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        default:
            TORCH_CHECK(false, "Unsupported hidden dim");
    }
}

at::Tensor rmsnorm_forward_fp16_fp32_fp32_indivisible_v2(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (x.size(1)) {
        case 128:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 2, 128, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 2, 128, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 2, 128, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 256:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 2, 256, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 2, 256, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 2, 256, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 512:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 2, 512, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 2, 512, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 2, 512, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        default:
            TORCH_CHECK(false, "Unsupported hidden dim");
    }
}

at::Tensor rmsnorm_forward_fp16_fp32_fp32_indivisible_v4(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (x.size(1)) {
        case 128:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 4, 128, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 4, 128, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 4, 128, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 256:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 4, 256, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 4, 256, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 4, 256, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 512:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 4, 512, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 4, 512, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 4, 512, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        default:
            TORCH_CHECK(false, "Unsupported hidden dim");
    }
}

at::Tensor rmsnorm_forward_fp16_fp32_fp32_indivisible_v8(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (x.size(1)) {
        case 128:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 8, 128, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 8, 128, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 8, 128, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 256:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 8, 256, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 8, 256, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 8, 256, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 512:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 8, 512, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 8, 512, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 8, 512, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        default:
            TORCH_CHECK(false, "Unsupported hidden dim");
    }
}

// BF16/BF16/FP32 (ACT=at::BFloat16, WGT=at::BFloat16, CALC=float)

at::Tensor rmsnorm_forward_bf16_bf16_fp32_indivisible_v1(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (x.size(1)) {
        case 128:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 128, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 128, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 128, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 256:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 256, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 256, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 256, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 512:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 512, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 512, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 512, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        default:
            TORCH_CHECK(false, "Unsupported hidden dim");
    }
}

at::Tensor rmsnorm_forward_bf16_bf16_fp32_indivisible_v2(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (x.size(1)) {
        case 128:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 128, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 128, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 128, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 256:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 256, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 256, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 256, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 512:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 512, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 512, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 512, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        default:
            TORCH_CHECK(false, "Unsupported hidden dim");
    }
}

at::Tensor rmsnorm_forward_bf16_bf16_fp32_indivisible_v4(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (x.size(1)) {
        case 128:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 128, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 128, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 128, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 256:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 256, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 256, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 256, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 512:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 512, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 512, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 512, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        default:
            TORCH_CHECK(false, "Unsupported hidden dim");
    }
}

at::Tensor rmsnorm_forward_bf16_bf16_fp32_indivisible_v8(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (x.size(1)) {
        case 128:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 128, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 128, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 128, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 256:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 256, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 256, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 256, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 512:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 512, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 512, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 512, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        default:
            TORCH_CHECK(false, "Unsupported hidden dim");
    }
}

// BF16/FP32/FP32 (ACT=at::BFloat16, WGT=float, CALC=float)

at::Tensor rmsnorm_forward_bf16_fp32_fp32_indivisible_v1(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (x.size(1)) {
        case 128:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 1, 128, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 1, 128, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 1, 128, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 256:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 1, 256, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 1, 256, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 1, 256, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 512:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 1, 512, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 1, 512, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 1, 512, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        default:
            TORCH_CHECK(false, "Unsupported hidden dim");
    }
}

at::Tensor rmsnorm_forward_bf16_fp32_fp32_indivisible_v2(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (x.size(1)) {
        case 128:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 2, 128, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 2, 128, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 2, 128, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 256:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 2, 256, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 2, 256, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 2, 256, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 512:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 2, 512, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 2, 512, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 2, 512, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        default:
            TORCH_CHECK(false, "Unsupported hidden dim");
    }
}

at::Tensor rmsnorm_forward_bf16_fp32_fp32_indivisible_v4(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (x.size(1)) {
        case 128:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 4, 128, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 4, 128, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 4, 128, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 256:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 4, 256, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 4, 256, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 4, 256, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 512:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 4, 512, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 4, 512, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 4, 512, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        default:
            TORCH_CHECK(false, "Unsupported hidden dim");
    }
}

at::Tensor rmsnorm_forward_bf16_fp32_fp32_indivisible_v8(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (x.size(1)) {
        case 128:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 8, 128, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 8, 128, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 8, 128, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 256:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 8, 256, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 8, 256, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 8, 256, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        case 512:
            switch (num_threads) {
                case 128:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 8, 512, 128>(x, gamma, epsilon);
                case 256:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 8, 512, 256>(x, gamma, epsilon);
                case 512:
                    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 8, 512, 512>(x, gamma, epsilon);
                default:
                    TORCH_CHECK(false, "Unsupported thread count");
            }
        default:
            TORCH_CHECK(false, "Unsupported hidden dim");
    }
}

// ============================================================================
// Indivisible example: DIM_X=130 (130 % 8 = 2, tail of 2 elements)
// ============================================================================

at::Tensor rmsnorm_forward_fp16_fp16_fp32_indivisible_130(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon
) {
    return cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 8, 130, 256>(x, gamma, epsilon);
}

// ============================================================================
// MASSIVE SCALE TEST LAUNCHERS (REMOVE AFTER TESTING)
// BF16/BF16/FP32, DIM=5120, VLCOUNT=1,2,4,8, THREADS=32,64,96,128,256,512,640,1024
// ============================================================================

at::Tensor rmsnorm_forward_bf16_bf16_fp32_v1_dim5120(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (num_threads) {
        case 32:
            return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 5120, 32>(x, gamma, epsilon);
        case 64:
            return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 5120, 64>(x, gamma, epsilon);
        case 96:
            return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 5120, 96>(x, gamma, epsilon);
        case 128:
            return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 5120, 128>(x, gamma, epsilon);
        case 256:
            return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 5120, 256>(x, gamma, epsilon);
        case 512:
            return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 5120, 512>(x, gamma, epsilon);
        case 640:
            return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 5120, 640>(x, gamma, epsilon);
        case 1024:
            return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 5120, 1024>(x, gamma, epsilon);
        default:
            TORCH_CHECK(false, "Unsupported thread count");
    }
}

at::Tensor rmsnorm_forward_bf16_bf16_fp32_v2_dim5120(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (num_threads) {
        case 32:
            return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 5120, 32>(x, gamma, epsilon);
        case 64:
            return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 5120, 64>(x, gamma, epsilon);
        case 96:
            return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 5120, 96>(x, gamma, epsilon);
        case 128:
            return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 5120, 128>(x, gamma, epsilon);
        case 256:
            return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 5120, 256>(x, gamma, epsilon);
        case 512:
            return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 5120, 512>(x, gamma, epsilon);
        case 640:
            return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 5120, 640>(x, gamma, epsilon);
        case 1024:
            return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 5120, 1024>(x, gamma, epsilon);
        default:
            TORCH_CHECK(false, "Unsupported thread count");
    }
}

at::Tensor rmsnorm_forward_bf16_bf16_fp32_v4_dim5120(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (num_threads) {
        case 32:
            return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 5120, 32>(x, gamma, epsilon);
        case 64:
            return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 5120, 64>(x, gamma, epsilon);
        case 96:
            return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 5120, 96>(x, gamma, epsilon);
        case 128:
            return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 5120, 128>(x, gamma, epsilon);
        case 256:
            return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 5120, 256>(x, gamma, epsilon);
        case 512:
            return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 5120, 512>(x, gamma, epsilon);
        case 640:
            return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 5120, 640>(x, gamma, epsilon);
        case 1024:
            return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 5120, 1024>(x, gamma, epsilon);
        default:
            TORCH_CHECK(false, "Unsupported thread count");
    }
}

at::Tensor rmsnorm_forward_bf16_bf16_fp32_v8_dim5120(
    const at::Tensor& x,
    const at::Tensor& gamma,
    float epsilon,
    int num_threads
) {
    switch (num_threads) {
        case 32:
            return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 5120, 32>(x, gamma, epsilon);
        case 64:
            return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 5120, 64>(x, gamma, epsilon);
        case 96:
            return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 5120, 96>(x, gamma, epsilon);
        case 128:
            return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 5120, 128>(x, gamma, epsilon);
        case 256:
            return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 5120, 256>(x, gamma, epsilon);
        case 512:
            return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 5120, 512>(x, gamma, epsilon);
        case 640:
            return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 5120, 640>(x, gamma, epsilon);
        case 1024:
            return cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 5120, 1024>(x, gamma, epsilon);
        default:
            TORCH_CHECK(false, "Unsupported thread count");
    }
}
