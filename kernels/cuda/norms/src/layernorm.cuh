// LayerNorm formula: y_i = gamma_i * (x_i - mean) / root(variance + eps) + Bias
// Variance
#pragma once

#include <torch/torch.h>
#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>

namespace norms {

namespace kernels {

    // VECTORIZED_LOAD_COUNT: number of elements per vectorized load (1, 2, 4, 8)
    // DIM_X must be divisible by VECTORIZED_LOAD_COUNT for this kernel
    template<typename ACTIVATION_DTYPE, typename WEIGHT_DTYPE, typename CALCULATION_DTYPE, int VECTORIZED_LOAD_COUNT, int DIM_X, int THREADS_PER_BLOCK>
    __global__ void layer_norm_forward_kernel_offline(
        const ACTIVATION_DTYPE* __restrict__ x, const WEIGHT_DTYPE* __restrict__ gamma, const WEIGHT_DTYPE* __restrict__ bias, ACTIVATION_DTYPE* __restrict__ y, float epsilon);
    
    template<typename ACTIVATION_DTYPE, typename WEIGHT_DTYPE, typename CALCULATION_DTYPE, int THREADS_PER_BLOCK>
    __global__ void layer_norm_forward_kernel_online(
        const ACTIVATION_DTYPE* __restrict__ x, const WEIGHT_DTYPE* __restrict__ gamma, const WEIGHT_DTYPE* __restrict__ bias, ACTIVATION_DTYPE* __restrict__ y, float epsilon, int dim_x);
}

namespace launchers {

    namespace offline {

        template<typename ACTIVATION_DTYPE, typename WEIGHT_DTYPE, typename CALCULATION_DTYPE, int VECTORIZED_LOAD_COUNT, int DIM_X, int THREADS_PER_BLOCK>
        at::Tensor layer_norm_forward_launcher_offline(const at::Tensor &x, const at::Tensor &gamma, const at::Tensor &bias, float epsilon);

    }

    namespace online {

        template<typename ACTIVATION_DTYPE, typename WEIGHT_DTYPE, typename CALCULATION_DTYPE, int THREADS_PER_BLOCK>
        at::Tensor layer_norm_forward_launcher_online(const at::Tensor &x, const at::Tensor &gamma, const at::Tensor &bias, float epsilon, int dim_x);

    }

}

}