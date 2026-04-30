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
    __global__ void rms_norm_kernel_forward_offline(
        const ACTIVATION_DTYPE* __restrict__ x, const WEIGHT_DTYPE* __restrict__ gamma, ACTIVATION_DTYPE* __restrict__ y, float epsilon);
    
    // VECTORIZED_LOAD_COUNT: number of elements per vectorized load (1, 2, 4, 8)
    // DIM_X must be divisible by VECTORIZED_LOAD_COUNT for this kernel
    // dL/dx_i = (dL/dy_i * gamma_i) / RMS - (x_i / (DIM_X * RMS^3) * SUM(dL/dy_j * gamma_j * x_j) )
    template<typename ACTIVATION_DTYPE, typename WEIGHT_DTYPE, typename CALCULATION_DTYPE, int VECTORIZED_LOAD_COUNT, int DIM_X, int THREADS_PER_BLOCK>
    __global__ void rms_norm_kernel_backward_offline_dx(
        const ACTIVATION_DTYPE* __restrict__ x, const WEIGHT_DTYPE* __restrict__ gamma, const ACTIVATION_DTYPE* __restrict__ dy, ACTIVATION_DTYPE* __restrict__ dx, float epsilon);
    

    template<typename ACTIVATION_DTYPE, typename WEIGHT_DTYPE, typename CALCULATION_DTYPE, int THREADS_PER_BLOCK>
    __global__ void rms_norm_kernel_forward_online(
        const ACTIVATION_DTYPE* __restrict__ x, const WEIGHT_DTYPE* __restrict__ gamma, ACTIVATION_DTYPE* __restrict__ y, float epsilon, int dim_x);
}

namespace launchers {

    namespace offline {

        template<typename ACTIVATION_DTYPE, typename WEIGHT_DTYPE, typename CALCULATION_DTYPE, int VECTORIZED_LOAD_COUNT, int DIM_X, int THREADS_PER_BLOCK>
        at::Tensor rms_norm_forward_launcher_offline(const at::Tensor &x, const at::Tensor &gamma, float epsilon);

    }

    namespace online {

        template<typename ACTIVATION_DTYPE, typename WEIGHT_DTYPE, typename CALCULATION_DTYPE, int THREADS_PER_BLOCK>
        at::Tensor rms_norm_forward_launcher_online(const at::Tensor &x, const at::Tensor &gamma, float epsilon, int dim_x);

    }

}

}
