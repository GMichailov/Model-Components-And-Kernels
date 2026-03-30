#pragma once

#include "../utils.cuh"
#include <torch/torch.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>

namespace norm {

namespace kernels {

    // Currently enforces DIM_X divisible by VECTORIZED_LOAD_TYPE.
    template<typename ACTIVATION_DTYPE, typename WEIGHT_DTYPE, typename VECTORIZED_LOAD_TYPE, typename CALCULATION_DTYPE, int DIM_X>
    __global__ void rmsnorm_kernel_divisible_forward(
        const ACTIVATION_DTYPE* __restrict__ x, const WEIGHT_DTYPE* __restrict__ gamma, ACTIVATION_DTYPE* __restrict__ y, float epsilon) 
    {
        static_assert((DIM_X * sizeof(ACTIVATION_DTYPE)) % sizeof(VECTORIZED_LOAD_TYPE) == 0, "Vectorized load datatype must be able to cleanly load a row of data.");
        static_assert(sizeof(VECTORIZED_LOAD_TYPE) % sizeof(ACTIVATION_DTYPE) == 0, "Vectorized load type must be a a multiple of the activation size.");
        static_assert(sizeof(VECTORIZED_LOAD_TYPE) >= sizeof(ACTIVATION_DTYPE), "Vectorized load type must be the same or larger than the activation type.");
        static_assert(sizeof(CALCULATION_DTYPE) >= sizeof(ACTIVATION_DTYPE), "Calculation datatype must be a larger or equal to in size to activation type.");
        constexpr int act_vec_loads = sizeof(VECTORIZED_LOAD_TYPE) / sizeof(ACTIVATION_DTYPE);
        constexpr int num_loads = DIM_X * sizeof(ACTIVATION_DTYPE) / sizeof(VECTORIZED_LOAD_TYPE);
        ACTIVATION_DTYPE x_vals[act_vec_loads];
        ACTIVATION_DTYPE out[act_vec_loads];
        const ACTIVATION_DTYPE* x_row = x + blockIdx.x * DIM_X;
        ACTIVATION_DTYPE* y_row = y + blockIdx.x * DIM_X;
        CALCULATION_DTYPE thread_sum{0};
        const VECTORIZED_LOAD_TYPE* x_vec = reinterpret_cast<const VECTORIZED_LOAD_TYPE*>(x_row);
        #pragma unroll
        for (int i{threadIdx.x}; i < num_loads; i += blockDim.x) {
            VECTORIZED_LOAD_TYPE v = x_vec[i];
            memcpy(x_vals, &v, sizeof(v));
            #pragma unroll
            for (int j{0}; j < act_vec_loads; ++j) {
                CALCULATION_DTYPE val = CALCULATION_DTYPE(x_vals[j]);
                thread_sum += val*val;
            }
        }
        CALCULATION_DTYPE sum = utils::block_reduce_sum<CALCULATION_DTYPE>(thread_sum);
        __shared__ CALCULATION_DTYPE inv_rms;
        if (threadIdx.x == 0) inv_rms = CALCULATION_DTYPE(rsqrtf(sum / DIM_X + epsilon));
        __syncthreads();
        VECTORIZED_LOAD_TYPE* y_vec = reinterpret_cast<VECTORIZED_LOAD_TYPE*>(y_row);
        for (int i{threadIdx.x}; i < num_loads; i += blockDim.x) {
            VECTORIZED_LOAD_TYPE x_vec_val = x_vec[i];
            memcpy(x_vals, &x_vec_val, sizeof(x_vec_val));
            #pragma unroll
            for (int j{0}; j < act_vec_loads; ++j) {
                CALCULATION_DTYPE weight = CALCULATION_DTYPE(gamma[j + i * act_vec_loads]);
                out[j] = ACTIVATION_DTYPE(CALCULATION_DTYPE(x_vals[j]) * weight * inv_rms);
            }
            memcpy(y_vec + i, out, sizeof(VECTORIZED_LOAD_TYPE));
        }
    }

    // Currently allows DIM_X to be indivisible by VECTORIZED_LOAD_TYPE.
    template<typename ACTIVATION_DTYPE, typename WEIGHT_DTYPE, typename VECTORIZED_LOAD_TYPE, typename CALCULATION_DTYPE, int DIM_X>
    __global__ void rmsnorm_kernel_indivisible_forward(
        const ACTIVATION_DTYPE* __restrict__ x, const WEIGHT_DTYPE* __restrict__ gamma, ACTIVATION_DTYPE* __restrict__ y, float epsilon) 
    {
        static_assert(sizeof(VECTORIZED_LOAD_TYPE) % sizeof(ACTIVATION_DTYPE) == 0, "Vectorized load type must be a a multiple of the activation size.");
        static_assert(sizeof(VECTORIZED_LOAD_TYPE) >= sizeof(ACTIVATION_DTYPE), "Vectorized load type must be the same or larger than the activation type.");
        static_assert(sizeof(CALCULATION_DTYPE) >= sizeof(ACTIVATION_DTYPE), "Calculation accumulation datatype must be a larger size than activation type.");
        constexpr int act_vec_loads = sizeof(VECTORIZED_LOAD_TYPE) / sizeof(ACTIVATION_DTYPE);
        constexpr int num_loads = DIM_X / act_vec_loads;
        constexpr int tail = DIM_X % act_vec_loads;
        int tail_idx = num_loads * act_vec_loads + threadIdx.x;
        ACTIVATION_DTYPE x_vals[act_vec_loads];
        ACTIVATION_DTYPE out[act_vec_loads];
        const ACTIVATION_DTYPE* x_row = x + blockIdx.x * DIM_X;
        ACTIVATION_DTYPE* y_row = y + blockIdx.x * DIM_X;
        CALCULATION_DTYPE thread_sum{0};
        const VECTORIZED_LOAD_TYPE* x_vec = reinterpret_cast<const VECTORIZED_LOAD_TYPE*>(x_row);
        #pragma unroll
        for (int i{threadIdx.x}; i < num_loads; i += blockDim.x) {
            VECTORIZED_LOAD_TYPE v = x_vec[i];
            memcpy(x_vals, &v, sizeof(v));
            #pragma unroll
            for (int j{0}; j < act_vec_loads; ++j) {
                CALCULATION_DTYPE val = CALCULATION_DTYPE(x_vals[j]);
                thread_sum += val*val;
            }
        }
        if(threadIdx.x < tail) { CALCULATION_DTYPE val = CALCULATION_DTYPE(x_row[tail_idx]); thread_sum += val * val; }
        CALCULATION_DTYPE sum = utils::block_reduce_sum<CALCULATION_DTYPE>(thread_sum);
        __shared__ CALCULATION_DTYPE inv_rms;
        if (threadIdx.x == 0) inv_rms = CALCULATION_DTYPE(rsqrtf(sum / DIM_X + epsilon));
        __syncthreads();
        VECTORIZED_LOAD_TYPE* y_vec = reinterpret_cast<VECTORIZED_LOAD_TYPE*>(y_row);
        for (int i{threadIdx.x}; i < num_loads; i += blockDim.x) {
            VECTORIZED_LOAD_TYPE x_vec_val = x_vec[i];
            memcpy(x_vals, &x_vec_val, sizeof(x_vec_val));
            #pragma unroll
            for (int j{0}; j < act_vec_loads; ++j) {
                CALCULATION_DTYPE weight = CALCULATION_DTYPE(gamma[j + i * act_vec_loads]);
                out[j] = ACTIVATION_DTYPE(CALCULATION_DTYPE(x_vals[j]) * weight * inv_rms);
            }
            memcpy(y_vec + i, out, sizeof(VECTORIZED_LOAD_TYPE));
        }
        if (threadIdx.x < tail) { y_row[tail_idx] = ACTIVATION_DTYPE(CALCULATION_DTYPE(x_row[tail_idx]) * CALCULATION_DTYPE(gamma[tail_idx]) * inv_rms); }
    }

}

namespace launchers {

    namespace offline {

        template<typename ACTIVATION_DTYPE, typename WEIGHT_DTYPE, typename VECTORIZED_LOAD_TYPE, typename CALCULATION_TYPE, int DIM_X>
        torch::Tensor cuda_rmsnorm_divisble_forward_launcher_offline(const torch::Tensor &x, const torch::Tensor &gamma, float epsilon, int num_threads) {
            torch::Tensor y = torch::empty_like(x);
            dim3 grid(x.numel() / DIM_X);
            dim3 block(num_threads);
            rmsnorm_kernel_divisible<ACTIVATION_DTYPE, WEIGHT_DTYPE, VECTORIZED_LOAD_TYPE, CALCULATION_TYPE, DIM_X><<<grid, block>>>(x.data_ptr<ACTIVATION_DTYPE>(), gamma.data_ptr<WEIGHT_DTYPE>(), y.data_ptr<ACTIVATION_DTYPE>(), epsilon);
            return y;
        }

        template<typename ACTIVATION_DTYPE, typename WEIGHT_DTYPE, typename VECTORIZED_LOAD_TYPE, typename CALCULATION_TYPE, int DIM_X>
        torch::Tensor cuda_rmsnorm_indivisble_forward_launcher_offline(const torch::Tensor &x, const torch::Tensor &gamma, float epsilon, int num_threads) {
            torch::Tensor y = torch::empty_like(x);
            dim3 grid(x.numel() / DIM_X);
            dim3 block(num_threads);
            rmsnorm_kernel_indivisible<ACTIVATION_DTYPE, WEIGHT_DTYPE, VECTORIZED_LOAD_TYPE, CALCULATION_TYPE, DIM_X><<<grid, block>>>(x.data_ptr<ACTIVATION_DTYPE>(), gamma.data_ptr<WEIGHT_DTYPE>(), y.data_ptr<ACTIVATION_DTYPE>(), epsilon);
            return y;
        }

    }

    namespace online {



    }

}

}