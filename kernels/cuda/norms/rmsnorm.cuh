#pragma once

#include "../utils.cuh"
#include <torch/torch.h>
#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>

namespace norms {

namespace kernels {

    // NOTE: The datatypes that stuff 2 of the same (ie half2 and __nv_bfloat162) are not being unpacked correctly.
    // Still not clear if even worth unpacking or simply avoiding.

    // Currently enforces DIM_X divisible by VECTORIZED_LOAD_TYPE.
    template<typename ACTIVATION_DTYPE, typename WEIGHT_DTYPE, typename VECTORIZED_LOAD_TYPE, typename CALCULATION_DTYPE, int DIM_X>
    __global__ void rmsnorm_kernel_divisible_forward(
        const ACTIVATION_DTYPE* __restrict__ x, const WEIGHT_DTYPE* __restrict__ gamma, ACTIVATION_DTYPE* __restrict__ y, float epsilon);

    // Currently allows DIM_X to be indivisible by VECTORIZED_LOAD_TYPE.
    template<typename ACTIVATION_DTYPE, typename WEIGHT_DTYPE, typename VECTORIZED_LOAD_TYPE, typename CALCULATION_DTYPE, int DIM_X>
    __global__ void rmsnorm_kernel_indivisible_forward(
        const ACTIVATION_DTYPE* __restrict__ x, const WEIGHT_DTYPE* __restrict__ gamma, ACTIVATION_DTYPE* __restrict__ y, float epsilon);

}

namespace launchers {

    namespace offline {

        template<typename ACTIVATION_DTYPE, typename WEIGHT_DTYPE, typename VECTORIZED_LOAD_TYPE, typename CALCULATION_TYPE, int DIM_X>
        at::Tensor cuda_rmsnorm_divisble_forward_launcher_offline(const at::Tensor &x, const at::Tensor &gamma, float epsilon, int num_threads);

        template<typename ACTIVATION_DTYPE, typename WEIGHT_DTYPE, typename VECTORIZED_LOAD_TYPE, typename CALCULATION_TYPE, int DIM_X>
        at::Tensor cuda_rmsnorm_indivisble_forward_launcher_offline(const at::Tensor &x, const at::Tensor &gamma, float epsilon, int num_threads);

    }

    namespace online {



    }

}

}
