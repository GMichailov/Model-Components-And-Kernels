#pragma once

#include "../../utils.cuh"
#include <torch/torch.h>
#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>

namespace norms {

namespace kernels {

    // VECTORIZED_LOAD_COUNT: number of elements per vectorized load (1, 2, 4, 8)
    // DIM_X must be divisible by VECTORIZED_LOAD_COUNT for this kernel
    template<typename ACTIVATION_DTYPE, typename WEIGHT_DTYPE, typename CALCULATION_DTYPE, int VECTORIZED_LOAD_COUNT, int DIM_X, int THREADS_PER_BLOCK>
    __global__ void rmsnorm_kernel_divisible_forward(
        const ACTIVATION_DTYPE* __restrict__ x, const WEIGHT_DTYPE* __restrict__ gamma, ACTIVATION_DTYPE* __restrict__ y, float epsilon);
    
    // Allows DIM_X to be indivisible by VECTORIZED_LOAD_COUNT * ACTIVATION DATATYPE
    template<typename ACTIVATION_DTYPE, typename WEIGHT_DTYPE, typename CALCULATION_DTYPE, int VECTORIZED_LOAD_COUNT, int DIM_X, int THREADS_PER_BLOCK>
    __global__ void rmsnorm_kernel_indivisible_forward(
        const ACTIVATION_DTYPE* __restrict__ x, const WEIGHT_DTYPE* __restrict__ gamma, ACTIVATION_DTYPE* __restrict__ y, float epsilon);
}

namespace launchers {

    namespace offline {

        template<typename ACTIVATION_DTYPE, typename WEIGHT_DTYPE, typename CALCULATION_DTYPE, int VECTORIZED_LOAD_COUNT, int DIM_X, int THREADS_PER_BLOCK>
        at::Tensor cuda_rmsnorm_divisible_forward_launcher_offline(const at::Tensor &x, const at::Tensor &gamma, float epsilon);

        template<typename ACTIVATION_DTYPE, typename WEIGHT_DTYPE, typename CALCULATION_DTYPE, int VECTORIZED_LOAD_COUNT, int DIM_X, int THREADS_PER_BLOCK>
        at::Tensor cuda_rmsnorm_indivisible_forward_launcher_offline(const at::Tensor &x, const at::Tensor &gamma, float epsilon);

    }

    namespace online {



    }

}

}
