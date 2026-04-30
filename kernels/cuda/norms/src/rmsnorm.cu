#include "rmsnorm.cuh"

// Important ones atm are fp32, half(fp16), and bf16 (__nv_bfloat16) for activation and calculation.

namespace norms {

namespace kernels {

    // Conversion helpers for bfloat16 (required when __CUDA_NO_BFLOAT16_CONVERSIONS__ is defined)
    __device__ __forceinline__ float to_float(__nv_bfloat16 val) { return __bfloat162float(val); }
    __device__ __forceinline__ __nv_bfloat16 to_bfloat16(float val) { return __float2bfloat16(val); }
    __device__ __forceinline__ float to_float(float val) { return val; }
    __device__ __forceinline__ float to_float(half val) { return __half2float(val); }
    __device__ __forceinline__ half to_half(float val) { return __float2half(val); }
    __device__ __forceinline__ half to_half(half val) { return val; }

    __device__ __forceinline__ float warp_reduce_sum(float val) {
        #pragma unroll
        for (int offset{16}; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        return val;
    }

    template<typename T>
    __device__ __forceinline__ T block_reduce_sum(T val) {
        __shared__ float warp_sums[32];
        int lane = threadIdx.x % 32;
        int warp = threadIdx.x / 32;
        float fp32_val = float(val);
        fp32_val = warp_reduce_sum(fp32_val);
        if (lane == 0) warp_sums[warp] = fp32_val;
        __syncthreads();
        fp32_val = (threadIdx.x < ((blockDim.x + 31) / 32)) ? warp_sums[lane] : 0.0f;
        if (warp == 0) fp32_val = warp_reduce_sum(fp32_val);
        return T(fp32_val);
    }

    template<typename scalar_t, int vec_size>
    struct alignas(sizeof(scalar_t) * vec_size) aligned_vector{ scalar_t val[vec_size]; };

    template<typename wgt_scalar_t, int vec_size>
    struct alignas(sizeof(wgt_scalar_t) * vec_size) aligned_wgt_vector{ wgt_scalar_t val[vec_size]; };

    // Currently enforces DIM_X divisible by VECTORIZED_LOAD_TYPE. Launcher will default to 1 for weird dims.
    template<typename ACTIVATION_DTYPE, typename WEIGHT_DTYPE, typename CALCULATION_DTYPE, int VECTORIZED_LOAD_COUNT, int DIM_X, int THREADS_PER_BLOCK>
    __global__ void rms_norm_kernel_forward_offline(
        const ACTIVATION_DTYPE* __restrict__ x, const WEIGHT_DTYPE* __restrict__ gamma, ACTIVATION_DTYPE* __restrict__ y, float epsilon)
    {
        using vec_t = aligned_vector<ACTIVATION_DTYPE, VECTORIZED_LOAD_COUNT>;
        using wgt_vec_t = aligned_wgt_vector<WEIGHT_DTYPE, VECTORIZED_LOAD_COUNT>;
        static_assert(VECTORIZED_LOAD_COUNT > 0, "Vectorized load count must be at least 1.");
        static_assert(THREADS_PER_BLOCK <= 1024, "Max threads per block is 1024.");
        static_assert(sizeof(vec_t) <= 16, "Max vectorized load size must be less than or equal to 128 bits/16 bytes.");
        static_assert(DIM_X % VECTORIZED_LOAD_COUNT == 0, "Vectorized load datatype must be able to cleanly load a row of data.");
        static_assert(sizeof(CALCULATION_DTYPE) >= sizeof(ACTIVATION_DTYPE), "Calculation datatype must be a larger or equal to in size to activation type.");
        // Add an assertion that bfloat16 and half are NOT used together.
        constexpr int num_loads = DIM_X / VECTORIZED_LOAD_COUNT;
        const ACTIVATION_DTYPE* x_row = x + blockIdx.x * DIM_X;
        ACTIVATION_DTYPE* y_row = y + blockIdx.x * DIM_X;
        CALCULATION_DTYPE thread_sum{0};
        const vec_t* x_vec = reinterpret_cast<const vec_t*>(x_row);
        const wgt_vec_t* gamma_vec = reinterpret_cast<const wgt_vec_t*>(gamma);
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
                thread_sum += val*val;
            }
        }
        CALCULATION_DTYPE inv_rms = block_reduce_sum<CALCULATION_DTYPE>(thread_sum);
        inv_rms = static_cast<CALCULATION_DTYPE>(rsqrtf(sum / DIM_X + epsilon));
        __syncthreads();
        vec_t* y_vec = reinterpret_cast<vec_t*>(y_row);
        #pragma unroll
        for (uint i{threadIdx.x}; i < num_loads; i += THREADS_PER_BLOCK) {
            vec_t x_vec_val = x_vec[i];
            wgt_vec_t gamma_vec_val = gamma_vec[i];
            vec_t out;
            #pragma unroll
            for (int j{0}; j < VECTORIZED_LOAD_COUNT; ++j) {
                // CALC_DTYPE >= ACT_DTYPE and bloat and half cannot be used simultaneously.
                if constexpr (std::is_same_v<ACTIVATION_DTYPE, float>) {
                    // Float and Float
                    out.val[j] = x_vec_val.val[j] * gamma_vec_val.val[j] * inv_rms;
                } else if constexpr (std::is_same_v<ACTIVATION_DTYPE, half>) {
                    if constexpr (std::is_same_v<CALCULATION_DTYPE, float>) {
                        out.val[j] = to_half(to_float(x_vec_val.val[j]) * to_float(gamma_vec_val.val[j]) * inv_rms);
                    } else if constexpr (std::is_same_v<CALCULATION_DTYPE, half>) {
                        out.val[j] = x_vec_val.val[j] * gamma_vec_val.val[j] * inv_rms;
                    }
                } else if constexpr (std::is_same_v<ACTIVATION_DTYPE, __nv_bfloat16>) {
                    if constexpr (std::is_same_v<CALCULATION_DTYPE, float>) {
                        out.val[j] = to_bfloat16(to_float(x_vec_val.val[j]) * to_float(gamma_vec_val.val[j]) * inv_rms);
                    } else if constexpr (std::is_same_v<CALCULATION_DTYPE, __nv_bfloat16>) {
                        out.val[j] = x_vec_val.val[j] * gamma_vec_val.val[j] * inv_rms;
                    }
                }
            }
            y_vec[i] = out;
        }
    }

    // VECTORIZED_LOAD_COUNT: number of elements per vectorized load (1, 2, 4, 8)
    // DIM_X must be divisible by VECTORIZED_LOAD_COUNT for this kernel
    // dL/dx_i = (dL/dy_i * gamma_i) / RMS - (x_i / (DIM_X * RMS^3) * SUM(dL/dy_j * gamma_j * x_j) )
    template<typename ACTIVATION_DTYPE, typename WEIGHT_DTYPE, typename CALCULATION_DTYPE, int VECTORIZED_LOAD_COUNT, int DIM_X, int THREADS_PER_BLOCK>
    __global__ void rms_norm_kernel_backward_offline_dx(
        const ACTIVATION_DTYPE* __restrict__ x, const WEIGHT_DTYPE* __restrict__ gamma, const ACTIVATION_DTYPE* __restrict__ dy, ACTIVATION_DTYPE* __restrict__ dx, float epsilon)
    {
        using vec_t = aligned_vector<ACTIVATION_DTYPE, VECTORIZED_LOAD_COUNT>;
        using wgt_vec_t = aligned_wgt_vector<WEIGHT_DTYPE, VECTORIZED_LOAD_COUNT>;
        static_assert(VECTORIZED_LOAD_COUNT > 0, "Vectorized load count must be at least 1.");
        static_assert(THREADS_PER_BLOCK <= 1024, "Max threads per block is 1024.");
        static_assert(sizeof(vec_t) <= 16, "Max vectorized load size must be less than or equal to 128 bits/16 bytes.");
        static_assert(DIM_X % VECTORIZED_LOAD_COUNT == 0, "Vectorized load datatype must be able to cleanly load a row of data.");
        static_assert(sizeof(CALCULATION_DTYPE) >= sizeof(ACTIVATION_DTYPE), "Calculation datatype must be a larger or equal to in size to activation type.");
        // Add an assertion that bfloat16 and half are NOT used together.
        constexpr int num_loads = DIM_X / VECTORIZED_LOAD_COUNT;
        const ACTIVATION_DTYPE* x_row = x + blockIdx.x * DIM_X;
        ACTIVATION_DTYPE* dy_row = dy + blockIdx.x * DIM_X;
        CALCULATION_DTYPE* thread_rms_sum{0};
        CALCULATION_DTYPE* thread_coeff_sum{0};
        const vec_t* x_vec = reinterpret_cast<const vec_t*>(x_row);
        const wgt_vec_t* gamma_vec = reinterpret_cast<const wgt_vec_t*>(gamma);
        const wgt_vec_t* dy_vec = reinterpret_cast<const vec_t*>(dy_row);
        #pragma unroll
        for (uint i{threadIdx.x}; i < num_loads; i += THREADS_PER_BLOCK) {
            vec_t x_vec_val = x_vec[i];
            wgt_vec_t gamma_vec_val = gamma_vec[i];
            vec_t dy_vec_val = dy_vec[i];
            #pragma unroll
            for (int j{0}; j < VECTORIZED_LOAD_COUNT; ++j) {
                CALCULATION_DTYPE x_val;
                CALCULATION_DTYPE gamma_val;
                CALCULATION_DTYPE dy_val;
                if constexpr (std::is_same_v<CALCULATION_DTYPE, float>) {
                    x_val = to_float(x_vec_val.val[j]);
                    gamma_val = to_float(gamma_vec_val.val[j]);
                    dy_val = to_float(dy_vec_val.val[j]);
                } else if constexpr (std::is_same_v<CALCULATION_DTYPE, __nv_bfloat16>) {
                    x_val = to_bfloat16(x_vec_val.val[j]);
                    gamma_val = to_bfloat16(gamma_vec_val.val[j]);
                    dy_val = to_bfloat16(dy_vec_val.val[j]);
                } else if constexpr (std::is_same_v<CALCULATION_DTYPE, half>) {
                    x_val = to_half(x_vec_val.val[j]);
                    gamma_val = to_half(gamma_vec_val.val[j]);
                    dy_val = to_half(dy_vec_val.val[j]);
                }
                thread_rms_sum += x_val * x_val;
                thread_coeff_sum += x_val * gamma_val * dy_val;
            }
        }
        CALCULATION_DTYPE block_inv_rms = block_reduce_sum<CALCULATION_DTYPE>(thread_rms_sum);
        block_inv_rms = static_cast<CALCULATION_DTYPE>(rsqrtf(block_rms_sum / DIM_X + epsilon));
        CALCULATION_DTYPE block_coeff = block_reduce_sum<CALCULATION_DTYPE>(thread_coeff_sum);
        block_coeff *= block_inv_rms * block_inv_rms * block_inv_rms * DIM_X;
        vec_t* dx_vec = reinterpret_cast<vec_t*>(dx + blockIdx.x * DIM_X);
        #pragma unroll
        for (uint i{threadIdx.x}; i < num_loads; i += THREADS_PER_BLOCK) {
            vec_t x_vec_val = x_vec[i];
            wgt_vec_t gamma_vec_val = gamma_vec[i];
            vec_t dy_vec_val = dy_vec[i];
            vec_t out;
            #pragma unroll
            for (int j{0}; j < VECTORIZED_LOAD_COUNT; ++j) {
                // CALC_DTYPE >= ACT_DTYPE and bloat and half cannot be used simultaneously.
                if constexpr (std::is_same_v<ACTIVATION_DTYPE, float>) {
                    // Float and Float
                    out.val[j] = dy_vec_val.val[j] * gamma_vec_val.val[j] * inv_rms - x_vec_val.val[j] * block_coeff;
                } else if constexpr (std::is_same_v<ACTIVATION_DTYPE, half>) {
                    if constexpr (std::is_same_v<CALCULATION_DTYPE, float>) {
                        out.val[j] = to_half(to_float(dy_vec_val.val[j] * gamma_vec_val.val[j]) * inv_rms - to_float(x_vec_val.val[j]) * block_coeff);
                    } else if constexpr (std::is_same_v<CALCULATION_DTYPE, half>) {
                        out.val[j] = dy_vec_val.val[j] * gamma_vec_val.val[j] * inv_rms - x_vec_val.val[j] * block_coeff;
                    }
                } else if constexpr (std::is_same_v<ACTIVATION_DTYPE, __nv_bfloat16>) {
                    if constexpr (std::is_same_v<CALCULATION_DTYPE, float>) {
                        out.val[j] = to_bfloat16(to_float(dy_vec_val.val[j] * gamma_vec_val.val[j]) * inv_rms - to_float(x_vec_val.val[j]) * block_coeff);
                    } else if constexpr (std::is_same_v<CALCULATION_DTYPE, __nv_bfloat16>) {
                        out.val[j] = dy_vec_val.val[j] * gamma_vec_val.val[j] * inv_rms - x_vec_val.val[j] * block_coeff;
                    }
                }
            }
            dx_vec[i] = out;
        }
    }

    // TODO: Write a quick kernel like the one above that launches 1 warp per 2 rows.


    template<typename ACTIVATION_DTYPE, typename WEIGHT_DTYPE, typename CALCULATION_DTYPE, int THREADS_PER_BLOCK>
    __global__ void rms_norm_kernel_forward_online(
        const ACTIVATION_DTYPE* __restrict__ x, const WEIGHT_DTYPE* __restrict__ gamma, ACTIVATION_DTYPE* __restrict__ y, float epsilon, int dim_x)
    {
        static_assert(THREADS_PER_BLOCK <= 1024, "Max threads per block is 1024.");
        static_assert(sizeof(CALCULATION_DTYPE) >= sizeof(ACTIVATION_DTYPE), "Calculation datatype must be a larger or equal to in size to activation type.");
        // Add an assertion that bfloat16 and half are NOT used together.
        const ACTIVATION_DTYPE* x_row = x + blockIdx.x * dim_x;
        ACTIVATION_DTYPE* y_row = y + blockIdx.x * dim_x;
        CALCULATION_DTYPE thread_sum{0};
        for (uint i{threadIdx.x}; i < dim_x; i += THREADS_PER_BLOCK) {
            CALCULATION_DTYPE val;
            if constexpr (std::is_same_v<CALCULATION_DTYPE, float>) {
                val = to_float(x_row[i]);
            } else if constexpr (std::is_same_v<CALCULATION_DTYPE, __nv_bfloat16>) {
                val = to_bfloat16(x_row[i]);
            } else if constexpr (std::is_same_v<CALCULATION_DTYPE, half>) {
                val = to_half(x_row[i]);
            }
            thread_sum += val*val;
        }
        CALCULATION_DTYPE inv_rms = block_reduce_sum<CALCULATION_DTYPE>(thread_sum);
        inv_rms = static_cast<CALCULATION_DTYPE>(rsqrtf(sum / dim_x + epsilon));
        __syncthreads();
        for (uint i{threadIdx.x}; i < dim_x; i += THREADS_PER_BLOCK) {
            ACTIVATION_DTYPE out;
            // CALC_DTYPE >= ACT_DTYPE and bloat and half cannot be used simultaneously.
            if constexpr (std::is_same_v<ACTIVATION_DTYPE, float>) {
                // Float and Float
                out = x_row[i] * gamma[i] * inv_rms;
            } else if constexpr (std::is_same_v<ACTIVATION_DTYPE, half>) {
                if constexpr (std::is_same_v<CALCULATION_DTYPE, float>) {
                    out = to_half(to_float(x_row[i]) * to_float(gamma[i]) * inv_rms);
                } else if constexpr (std::is_same_v<CALCULATION_DTYPE, half>) {
                    out = x_row[i] * gamma[i] * inv_rms;
                }
            } else if constexpr (std::is_same_v<ACTIVATION_DTYPE, __nv_bfloat16>) {
                if constexpr (std::is_same_v<CALCULATION_DTYPE, float>) {
                    out = to_bfloat16(to_float(x_row[i]) * to_float(gamma[i]) * inv_rms);
                } else if constexpr (std::is_same_v<CALCULATION_DTYPE, __nv_bfloat16>) {
                    out = x_row[i] * gamma[i] * inv_rms;
                }
            }
            y_row[i] = out;
        }
    }

}

template<typename T>
struct torch_to_raw {
    using type = T;
};

template<>
struct torch_to_raw<at::BFloat16> {
    using type = __nv_bfloat16;
};

template<>
struct torch_to_raw<at::Half> {
    using type = half;
};

template<>
struct torch_to_raw<float> {
    using type = float;
};

namespace launchers {

    namespace offline {

        template<typename ACTIVATION_DTYPE, typename WEIGHT_DTYPE, typename CALCULATION_DTYPE, int VECTORIZED_LOAD_COUNT, int DIM_X, int THREADS_PER_BLOCK>
        at::Tensor rms_norm_forward_launcher_offline(const at::Tensor &x, const at::Tensor &gamma, float epsilon) {
            using cuda_act_dtype = typename torch_to_raw<ACTIVATION_DTYPE>::type;
            using cuda_wgt_dtype = typename torch_to_raw<WEIGHT_DTYPE>::type;
            using cuda_calc_dtype = typename torch_to_raw<CALCULATION_DTYPE>::type;
            using vec_t = norms::kernels::aligned_vector<cuda_act_dtype, VECTORIZED_LOAD_COUNT>;
            using wgt_vec_t = norms::kernels::aligned_wgt_vector<cuda_wgt_dtype, VECTORIZED_LOAD_COUNT>;
            TORCH_CHECK(x.size(1) == DIM_X, "Expected hidden dim ", DIM_X, " but got ", x.size(1));
            TORCH_CHECK(gamma.numel() == DIM_X, "Expected gamma len ", DIM_X, " but got ", gamma.numel());
            at::Tensor y = at::empty_like(x);
            const auto* x_ptr = reinterpret_cast<const cuda_act_dtype*>(x.data_ptr<ACTIVATION_DTYPE>());
            const auto* gamma_ptr = reinterpret_cast<const cuda_wgt_dtype*>(gamma.data_ptr<WEIGHT_DTYPE>());
            auto* y_ptr = reinterpret_cast<cuda_act_dtype*>(y.data_ptr<ACTIVATION_DTYPE>());
            TORCH_CHECK(reinterpret_cast<uintptr_t>(x_ptr) % alignof(vec_t) == 0, "x is not properly aligned for vectorized access");
            TORCH_CHECK(reinterpret_cast<uintptr_t>(gamma_ptr) % alignof(wgt_vec_t) == 0, "gamma is not properly aligned for vectorized access");
            TORCH_CHECK(reinterpret_cast<uintptr_t>(y_ptr) % alignof(vec_t) == 0, "y is not properly aligned for vectorized access");
            dim3 grid(x.numel() / DIM_X);
            dim3 block(THREADS_PER_BLOCK);
            norms::kernels::rms_norm_kernel_forward_offline<cuda_act_dtype, cuda_wgt_dtype, cuda_calc_dtype, VECTORIZED_LOAD_COUNT, DIM_X, THREADS_PER_BLOCK><<<grid, block>>>(x_ptr, gamma_ptr, y_ptr, epsilon);
            return y;
        }
    }

    namespace online {

        template<typename ACTIVATION_DTYPE, typename WEIGHT_DTYPE, typename CALCULATION_DTYPE, int THREADS_PER_BLOCK>
        at::Tensor cuda_rms_norm_forward_launcher_online(const at::Tensor &x, const at::Tensor &gamma, float epsilon, int dim_x) {
            using cuda_act_dtype = typename torch_to_raw<ACTIVATION_DTYPE>::type;
            using cuda_wgt_dtype = typename torch_to_raw<WEIGHT_DTYPE>::type;
            using cuda_calc_dtype = typename torch_to_raw<CALCULATION_DTYPE>::type;
            TORCH_CHECK(gamma.numel() == dim_x, "Expected gamma len ", dim_x, " but got ", gamma.numel());
            at::Tensor y = at::empty_like(x);
            const auto* x_ptr = reinterpret_cast<const cuda_act_dtype*>(x.data_ptr<ACTIVATION_DTYPE>());
            const auto* gamma_ptr = reinterpret_cast<const cuda_wgt_dtype*>(gamma.data_ptr<WEIGHT_DTYPE>());
            auto* y_ptr = reinterpret_cast<cuda_act_dtype*>(y.data_ptr<ACTIVATION_DTYPE>());
            dim3 grid(x.numel() / dim_x);
            dim3 block(THREADS_PER_BLOCK);
            norms::kernels::rms_norm_kernel_forward_online<cuda_act_dtype, cuda_wgt_dtype, cuda_calc_dtype, THREADS_PER_BLOCK><<<grid, block>>>(x_ptr, gamma_ptr, y_ptr, epsilon, dim_x);
            return y;
        }

    }

}

}

template at::Tensor norms::launchers::online::cuda_rms_norm_forward_launcher_online<at::BFloat16, at::BFloat16, float, 32>(const at::Tensor&, const at::Tensor&, float, int);




