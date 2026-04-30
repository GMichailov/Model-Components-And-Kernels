#include <torch/extension.h>
#include "layernorm.cuh"

namespace norms {

namespace kernels {

    // VECTORIZED_LOAD_COUNT: number of elements per vectorized load (1, 2, 4, 8)
    // DIM_X must be divisible by VECTORIZED_LOAD_COUNT for this kernel
    template<typename ACTIVATION_DTYPE, typename WEIGHT_DTYPE, typename CALCULATION_DTYPE, int VECTORIZED_LOAD_COUNT, int DIM_X, int THREADS_PER_BLOCK>
    __global__ void layer_norm_forward_kernel_offline(
        const ACTIVATION_DTYPE* __restrict__ x, const WEIGHT_DTYPE* __restrict__ gamma, const WEIGHT_DTYPE* __restrict__ bias, ACTIVATION_DTYPE* __restrict__ y, float epsilon) 
    {
        using vec_t = aligned_vector<ACTIVATION_DTYPE, VECTORIZED_LOAD_COUNT>;
        using wgt_vec_t = aligned_wgt_vector<WEIGHT_DTYPE, VECTORIZED_LOAD_COUNT>;
        static_assert(VECTORIZED_LOAD_COUNT > 0, "Vectorized load count must be at least 1.");
        static_assert(THREADS_PER_BLOCK <= 1024, "Max threads per block is 1024.");
        static_assert(sizeof(vec_t) <= 16, "Max vectorized load size must be less than or equal to 128 bits/16 bytes.");
        static_assert(DIM_X % VECTORIZED_LOAD_COUNT == 0, "Vectorized load datatype must be able to cleanly load a row of data.");
        static_assert(sizeof(CALCULATION_DTYPE) >= sizeof(ACTIVATION_DTYPE), "Calculation datatype must be a larger or equal to in size to activation type.");
        constexpr int num_loads = DIM_X / VECTORIZED_LOAD_COUNT;
        const ACTIVATION_DTYPE* x_row = x + blockIdx.x * DIM_X;
        ACTIVATION_DTYPE* y_row = y + blockIdx.x * DIM_X;
        CALCULATION_DTYPE thread_sum{0};
        const vec_t* x_vec = reinterpret_cast<const vec_t*>(x_row);
        vec_t* y_vec = reinterpret_cast<vec_t*>(y_row);
        const wgt_vec_t* gamma_vec = reinterpret_cast<const wgt_vec_t*>(gamma);
        const wgt_vec_t* bias_vec = reinterpret_cast<const wgt_vec_t*>(bias);
        // Calculate Mean
        #pragma unroll
        for (uint i{threadIdx.x}; i < num_loads; i += THREADS_PER_BLOCK) {
            vec_t v = x_vec[i];
            #pragma unroll
            for (int j{0}; j < VECTORIZED_LOAD_COUNT; ++j) {
                CALCULATION_DTYPE val;
                if constexpr (std::is_same_v<CALCULATION_DTYPE, float>) {
                    val = to_float(v.val[j]);
                } else if constexpr (std::is_same_v<CALCULATION_DTYPE, __nv_bfloat16>) {
                    val = to_bfloat16(v.val[j]);
                } else if constexpr (std::is_same_v<CALCULATION_DTYPE, half>) {
                    val = to_half(v.val[j]);
                }
                thread_sum += val;
            }
        }
        CALCULATION_DTYPE mean = block_reduce_sum<CALCULATION_DTYPE>(thread_sum);
        mean = static_cast<CALCULATION_DTYPE>(sum / DIM_X);
        // Calculate Standard Deviation
        thread_sum = 0;
        #pragma unroll
        for (uint i{threadIdx.x}; i < num_loads; i += THREADS_PER_BLOCK) {
            vec_t v = x_vec[i];
            #pragma unroll
            for (int j{0}; j < VECTORIZED_LOAD_COUNT; ++j) {
                CALCULATION_DTYPE val;
                if constexpr (std::is_same_v<CALCULATION_DTYPE, float>) {
                    val = to_float(v.val[j]) - mean;
                } else if constexpr (std::is_same_v<CALCULATION_DTYPE, __nv_bfloat16>) {
                    val = to_bfloat16(v.val[j]) - mean;
                } else if constexpr (std::is_same_v<CALCULATION_DTYPE, half>) {
                    val = to_half(v.val[j]) - mean; 
                }
                thread_sum += val * val;
            }
        }
        CALCULATION_DTYPE std_dev = block_reduce_sum<CALCULATION_DTYPE>(thread_sum);
        std_dev = static_cast<CALCULATION_DTYPE>(rsqrtf(sum / DIM_X + epsilon));
        // Calculate normalized values.
        #pragma unroll
        for (uint i{threadIdx.x}; i < num_loads; i += THREADS_PER_BLOCK) {
            vec_t x_vec_val = x_vec[i];
            wgt_vec_t gamma_vec_val = gamma_vec[i];
            wgt_vec_t bias_vec_val = bias_vec[i];
            vec_t out;
            #pragma unroll
            for (int j{0}; j < VECTORIZED_LOAD_COUNT; ++j) {
                // CALC_DTYPE >= ACT_DTYPE and bloat and half cannot be used simultaneously.
                if constexpr (std::is_same_v<ACTIVATION_DTYPE, float>) {
                    // Float and Float
                    out.val[j] = gamma_vec_val.val[j] * (x_vec_val.val[j] - mean) / std_dev + bias_vec_val.val[j];
                } else if constexpr (std::is_same_v<ACTIVATION_DTYPE, half>) {
                    if constexpr (std::is_same_v<CALCULATION_DTYPE, float>) {
                        out.val[j] = to_half(to_float(gamma_vec_val.val[j]) * to_float((x_vec_val.val[j] - mean) / std_dev + bias_vec_val.val[j]));
                    } else if constexpr (std::is_same_v<CALCULATION_DTYPE, half>) {
                        out.val[j] = gamma_vec_val.val[j] * (x_vec_val.val[j] - mean) / std_dev + bias_vec_val.val[j];
                    }
                } else if constexpr (std::is_same_v<ACTIVATION_DTYPE, __nv_bfloat16>) {
                    if constexpr (std::is_same_v<CALCULATION_DTYPE, float>) {
                        out.val[j] = to_bfloat16(to_float(gamma_vec_val.val[j]) * to_float((x_vec_val.val[j] - mean) / std_dev + bias_vec_val.val[j]));
                    } else if constexpr (std::is_same_v<CALCULATION_DTYPE, __nv_bfloat16>) {
                        out.val[j] = gamma_vec_val.val[j] * (x_vec_val.val[j] - mean) / std_dev + bias_vec_val.val[j];
                    }
                }
            }
            y_vec[i] = out;
        }
    }

    template<typename ACTIVATION_DTYPE, typename WEIGHT_DTYPE, typename CALCULATION_DTYPE, int THREADS_PER_BLOCK>
    __global__ void layer_norm_forward_kernel_online(
        const ACTIVATION_DTYPE* __restrict__ x, const WEIGHT_DTYPE* __restrict__ gamma, const WEIGHT_DTYPE* __restrict__ bias, ACTIVATION_DTYPE* __restrict__ y, float epsilon, int dim_x) 
    {
        static_assert(THREADS_PER_BLOCK <= 1024, "Max threads per block is 1024.");
        static_assert(sizeof(CALCULATION_DTYPE) >= sizeof(ACTIVATION_DTYPE), "Calculation datatype must be a larger or equal to in size to activation type.");
        const ACTIVATION_DTYPE* x_row = x + blockIdx.x * dim_x;
        ACTIVATION_DTYPE* y_row = y + blockIdx.x * dim_x;
        CALCULATION_DTYPE thread_sum{0};
        // Calculate Mean
        for (uint i{threadIdx.x}; i < dim_x; i += THREADS_PER_BLOCK) {
            CALCULATION_DTYPE val;
            if constexpr (std::is_same_v<CALCULATION_DTYPE, float>) {
                val = to_float(x_row[i]);
            } else if constexpr (std::is_same_v<CALCULATION_DTYPE, __nv_bfloat16>) {
                val = to_bfloat16(x_row[i]);
            } else if constexpr (std::is_same_v<CALCULATION_DTYPE, half>) {
                val = to_half(x_row[i]);
            }
            thread_sum += val;
        }
        CALCULATION_DTYPE mean = block_reduce_sum<CALCULATION_DTYPE>(thread_sum);
        mean = static_cast<CALCULATION_DTYPE>(sum / dim_x);
        // Calculate Standard Deviation
        thread_sum = 0;
        for (uint i{threadIdx.x}; i < dim_x; i += THREADS_PER_BLOCK) {
            CALCULATION_DTYPE val;
            if constexpr (std::is_same_v<CALCULATION_DTYPE, float>) {
                val = to_float(x_row[i]) - mean;
            } else if constexpr (std::is_same_v<CALCULATION_DTYPE, __nv_bfloat16>) {
                val = to_bfloat16(x_row[i]) - mean;
            } else if constexpr (std::is_same_v<CALCULATION_DTYPE, half>) {
                val = to_half(x_row[i]) - mean;
            }
            thread_sum += val * val;
        }
        CALCULATION_DTYPE std_dev = block_reduce_sum<CALCULATION_DTYPE>(thread_sum);
        std_dev = static_cast<CALCULATION_DTYPE>(rsqrtf(sum / dim_x + epsilon));
        // Calculate normalized values.
        for (uint i{threadIdx.x}; i < dim_x; i += THREADS_PER_BLOCK) {
            // CALC_DTYPE >= ACT_DTYPE and bloat and half cannot be used simultaneously.
            if constexpr (std::is_same_v<ACTIVATION_DTYPE, float>) {
                // Float and Float
                y_row[i] = gamma[i] * (x_row[i] - mean) / std_dev + bias[i];
            } else if constexpr (std::is_same_v<ACTIVATION_DTYPE, half>) {
                if constexpr (std::is_same_v<CALCULATION_DTYPE, float>) {
                    y_row[i] = to_half(to_float(gamma[i]) * to_float((x_row[i] - mean) / std_dev + bias[i]));
                } else if constexpr (std::is_same_v<CALCULATION_DTYPE, half>) {
                    y_row[i] = gamma[i] * (x_row[i] - mean) / std_dev + bias[i];
                }
            } else if constexpr (std::is_same_v<ACTIVATION_DTYPE, __nv_bfloat16>) {
                if constexpr (std::is_same_v<CALCULATION_DTYPE, float>) {
                    y_row[i] = to_bfloat16(to_float(gamma[i]) * to_float((x_row[i] - mean) / std_dev + bias[i]));
                } else if constexpr (std::is_same_v<CALCULATION_DTYPE, __nv_bfloat16>) {
                    y_row[i] = gamma[i] * (x_row[i] - mean) / std_dev + bias[i];
                }
            }
        }
    }

}

namespace launchers {

    namespace offline {
        
        template<typename ACTIVATION_DTYPE, typename WEIGHT_DTYPE, typename CALCULATION_DTYPE, int VECTORIZED_LOAD_COUNT, int DIM_X, int THREADS_PER_BLOCK>
        at::Tensor layer_norm_forward_launcher_offline(const at::Tensor &x, const at::Tensor &gamma, const at::Tensor &bias, float epsilon) {
            using cuda_act_dtype = typename torch_to_raw<ACTIVATION_DTYPE>::type;
            using cuda_wgt_dtype = typename torch_to_raw<WEIGHT_DTYPE>::type;
            using cuda_calc_dtype = typename torch_to_raw<CALCULATION_DTYPE>::type;
            using vec_t = norms::kernels::aligned_vector<cuda_act_dtype, VECTORIZED_LOAD_COUNT>;
            using wgt_vec_t = norms::kernels::aligned_wgt_vector<cuda_wgt_dtype, VECTORIZED_LOAD_COUNT>;
            TORCH_CHECK(x.size(1) == DIM_X, "Expected hidden dim ", DIM_X, " but got ", x.size(1));
            TORCH_CHECK(gamma.numel() == DIM_X, "Expected gamma len ", DIM_X, " but got ", gamma.numel());
            TORCH_CHECK(bias.numel() == DIM_X, "Expected bias len ", DIM_X, " but got ", bias.numel());
            at::Tensor y = at::empty_like(x);
            const auto* x_ptr = reinterpret_cast<const cuda_act_dtype*>(x.data_ptr<ACTIVATION_DTYPE>());
            const auto* gamma_ptr = reinterpret_cast<const cuda_wgt_dtype*>(gamma.data_ptr<WEIGHT_DTYPE>());
            const auto* bias_ptr = reinterpret_cast<const cuda_wgt_dtype*>(bias.data_ptr<WEIGHT_DTYPE>());
            auto* y_ptr = reinterpret_cast<cuda_act_dtype*>(y.data_ptr<ACTIVATION_DTYPE>());
            TORCH_CHECK(reinterpret_cast<uintptr_t>(x_ptr) % alignof(vec_t) == 0, "x is not properly aligned for vectorized access");
            TORCH_CHECK(reinterpret_cast<uintptr_t>(gamma_ptr) % alignof(wgt_vec_t) == 0, "gamma is not properly aligned for vectorized access");
            TORCH_CHECK(reinterpret_cast<uintptr_t>(bias_ptr) % alignof(wgt_vec_t) == 0, "bias is not properly aligned for vectorized access");
            TORCH_CHECK(reinterpret_cast<uintptr_t>(y_ptr) % alignof(vec_t) == 0, "y is not properly aligned for vectorized access");
            dim3 grid(x.numel() / DIM_X);
            dim3 block(THREADS_PER_BLOCK);
            norms::kernels::layer_norm_forward_kernel_offline<cuda_act_dtype, cuda_wgt_dtype, cuda_calc_dtype, VECTORIZED_LOAD_COUNT, DIM_X, THREADS_PER_BLOCK><<<grid, block>>>(x_ptr, gamma_ptr, bias_ptr, y_ptr, epsilon);
            return y;
        }
        
    }

    namespace online {

        template<typename ACTIVATION_DTYPE, typename WEIGHT_DTYPE, typename CALCULATION_DTYPE, int THREADS_PER_BLOCK>
        at::Tensor layer_norm_forward_launcher_online(const at::Tensor &x, const at::Tensor &gamma, const at::Tensor &bias, float epsilon, int dim_x) {
            using cuda_act_dtype = typename torch_to_raw<ACTIVATION_DTYPE>::type;
            using cuda_wgt_dtype = typename torch_to_raw<WEIGHT_DTYPE>::type;
            using cuda_calc_dtype = typename torch_to_raw<CALCULATION_DTYPE>::type;
            TORCH_CHECK(gamma.numel() == dim_x, "Expected gamma len ", dim_x, " but got ", gamma.numel());
            TORCH_CHECK(bias.numel() == dim_x, "Expected bias len ", dim_x, " but got ", gamma.numel());
            at::Tensor y = at::empty_like(x);
            const auto* x_ptr = reinterpret_cast<const cuda_act_dtype*>(x.data_ptr<ACTIVATION_DTYPE>());
            const auto* gamma_ptr = reinterpret_cast<const cuda_wgt_dtype*>(gamma.data_ptr<WEIGHT_DTYPE>());
            const auto* bias_ptr = reinterpret_cast<const cuda_wgt_dtype*>(bias.data_ptr<WEIGHT_DTYPE>());
            auto* y_ptr = reinterpret_cast<cuda_act_dtype*>(y.data_ptr<ACTIVATION_DTYPE>());
            dim3 grid(x.numel() / dim_x);
            dim3 block(THREADS_PER_BLOCK);
            norms::kernels::layer_norm_forward_kernel_online<cuda_act_dtype, cuda_wgt_dtype, cuda_calc_dtype, THREADS_PER_BLOCK><<<grid, block>>>(x_ptr, gamma_ptr, y_ptr, bias_ptr, epsilon, dim_x);
            return y;
        }

    }

}

}