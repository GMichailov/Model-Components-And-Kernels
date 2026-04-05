#include <torch/extension.h>
#include "rmsnorm.cuh"

// Important ones atm are fp32, half(fp16), and bf16 (__nv_bfloat16) for activation and calculation.

namespace norms {

namespace kernels {

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

    // NOTE: The datatypes that stuff 2 of the same (ie half2 and __nv_bfloat162) are not being unpacked correctly.
    // Still not clear if even worth unpacking or simply avoiding.

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
        for (uint i{threadIdx.x}; i < num_loads; i += blockDim.x) {
            VECTORIZED_LOAD_TYPE v = x_vec[i];
            memcpy(x_vals, &v, sizeof(v));
            #pragma unroll
            for (int j{0}; j < act_vec_loads; ++j) {
                CALCULATION_DTYPE val = static_cast<CALCULATION_DTYPE>(x_vals[j]);
                thread_sum += val*val;
            }
        }
        CALCULATION_DTYPE sum = block_reduce_sum<CALCULATION_DTYPE>(thread_sum);
        __shared__ CALCULATION_DTYPE inv_rms;
        if (threadIdx.x == 0) inv_rms = static_cast<CALCULATION_DTYPE>(rsqrtf(sum / DIM_X + epsilon));
        __syncthreads();
        VECTORIZED_LOAD_TYPE* y_vec = reinterpret_cast<VECTORIZED_LOAD_TYPE*>(y_row);
        for (uint i{threadIdx.x}; i < num_loads; i += blockDim.x) {
            VECTORIZED_LOAD_TYPE x_vec_val = x_vec[i];
            memcpy(x_vals, &x_vec_val, sizeof(x_vec_val));
            #pragma unroll
            for (int j{0}; j < act_vec_loads; ++j) {
                CALCULATION_DTYPE weight = static_cast<CALCULATION_DTYPE>(gamma[j + i * act_vec_loads]);
                out[j] = static_cast<ACTIVATION_DTYPE>(static_cast<CALCULATION_DTYPE>(x_vals[j]) * weight * inv_rms);
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
        for (uint i{threadIdx.x}; i < num_loads; i += blockDim.x) {
            VECTORIZED_LOAD_TYPE v = x_vec[i];
            memcpy(x_vals, &v, sizeof(v));
            #pragma unroll
            for (int j{0}; j < act_vec_loads; ++j) {
                CALCULATION_DTYPE val = static_cast<CALCULATION_DTYPE>(x_vals[j]);
                thread_sum += val*val;
            }
        }
        if(threadIdx.x < tail) { CALCULATION_DTYPE val = static_cast<CALCULATION_DTYPE>(x_row[tail_idx]); thread_sum += val * val; }
        CALCULATION_DTYPE sum = block_reduce_sum<CALCULATION_DTYPE>(thread_sum);
        __shared__ CALCULATION_DTYPE inv_rms;
        if (threadIdx.x == 0) inv_rms = static_cast<CALCULATION_DTYPE>(rsqrtf(sum / DIM_X + epsilon));
        __syncthreads();
        VECTORIZED_LOAD_TYPE* y_vec = reinterpret_cast<VECTORIZED_LOAD_TYPE*>(y_row);
        for (uint i{threadIdx.x}; i < num_loads; i += blockDim.x) {
            VECTORIZED_LOAD_TYPE x_vec_val = x_vec[i];
            memcpy(x_vals, &x_vec_val, sizeof(x_vec_val));
            #pragma unroll
            for (int j{0}; j < act_vec_loads; ++j) {
                CALCULATION_DTYPE weight = static_cast<CALCULATION_DTYPE>(gamma[j + i * act_vec_loads]);
                out[j] = static_cast<ACTIVATION_DTYPE>(static_cast<CALCULATION_DTYPE>(x_vals[j]) * weight * inv_rms);
            }
            memcpy(y_vec + i, out, sizeof(VECTORIZED_LOAD_TYPE));
        }
        if (threadIdx.x < tail) { y_row[tail_idx] = static_cast<ACTIVATION_DTYPE>(static_cast<CALCULATION_DTYPE>(x_row[tail_idx]) * static_cast<CALCULATION_DTYPE>(gamma[tail_idx]) * inv_rms); }
    }

}

namespace launchers {

    namespace offline {

        template<typename ACTIVATION_DTYPE, typename WEIGHT_DTYPE, typename VECTORIZED_LOAD_TYPE, typename CALCULATION_TYPE, int DIM_X>
        at::Tensor cuda_rmsnorm_divisble_forward_launcher_offline(const at::Tensor &x, const at::Tensor &gamma, float epsilon, int num_threads) {
            at::Tensor y = at::empty_like(x);
            dim3 grid(x.numel() / DIM_X);
            dim3 block(num_threads);
            kernels::rmsnorm_kernel_divisible_forward<ACTIVATION_DTYPE, WEIGHT_DTYPE, VECTORIZED_LOAD_TYPE, CALCULATION_TYPE, DIM_X><<<grid, block>>>(x.data_ptr<ACTIVATION_DTYPE>(), gamma.data_ptr<WEIGHT_DTYPE>(), y.data_ptr<ACTIVATION_DTYPE>(), epsilon);
            return y;
        }

        template<typename ACTIVATION_DTYPE, typename WEIGHT_DTYPE, typename VECTORIZED_LOAD_TYPE, typename CALCULATION_TYPE, int DIM_X>
        at::Tensor cuda_rmsnorm_indivisble_forward_launcher_offline(const at::Tensor &x, const at::Tensor &gamma, float epsilon, int num_threads) {
            at::Tensor y = at::empty_like(x);
            dim3 grid(x.numel() / DIM_X);
            dim3 block(num_threads);
            kernels::rmsnorm_kernel_indivisible_forward<ACTIVATION_DTYPE, WEIGHT_DTYPE, VECTORIZED_LOAD_TYPE, CALCULATION_TYPE, DIM_X><<<grid, block>>>(x.data_ptr<ACTIVATION_DTYPE>(), gamma.data_ptr<WEIGHT_DTYPE>(), y.data_ptr<ACTIVATION_DTYPE>(), epsilon);
            return y;
        }

    }

    namespace online {



    }

}

}

// ---------- TEMPLATE INSTANTIATIONS ----------
#define FOR_EACH_DIM(MACRO) \
    MACRO(128) /*\
    MACRO(256) \
    MACRO(384) \
    MACRO(512) \
    MACRO(768) \
    MACRO(1024) \
    MACRO(1536) \
    MACRO(2048) \
    MACRO(5120) \
    MACRO(10240) */

#define INSTANTIATE(ACT, WGT, VEC, CALC, DIM) \
    template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisble_forward_launcher_offline<ACT, WGT, VEC, CALC, DIM>(const at::Tensor&, const at::Tensor&, float, int);

// Use ATen datatypes for the templates.
// fp32, fp32, float4
#define INSTANTIATE_FP32_FP32_F4(DIM) INSTANTIATE(float, float, float4, float, DIM)
FOR_EACH_DIM(INSTANTIATE_FP32_FP32_F4)

// fp32, fp32, float2
#define INSTANTIATE_FP32_FP32_F2(DIM) INSTANTIATE(float, float, float2, float, DIM)
FOR_EACH_DIM(INSTANTIATE_FP32_FP32_F2)

// fp32, fp32, float
#define INSTANTIATE_FP32_FP32_F(DIM) INSTANTIATE(float, float, float, float, DIM)
FOR_EACH_DIM(INSTANTIATE_FP32_FP32_F)

// fp16, fp16, float4
#define INSTANTIATE_FP16_FP16_F4(DIM) INSTANTIATE(at::Half, at::Half, float4, float, DIM)
FOR_EACH_DIM(INSTANTIATE_FP16_FP16_F4)

// fp16, fp16, float2
#define INSTANTIATE_FP16_FP16_F2(DIM) INSTANTIATE(at::Half, at::Half, float2, float, DIM)
FOR_EACH_DIM(INSTANTIATE_FP16_FP16_F2)

// fp16, fp16, float
#define INSTANTIATE_FP16_FP16_F(DIM) INSTANTIATE(at::Half, at::Half, float, float, DIM)
FOR_EACH_DIM(INSTANTIATE_FP16_FP16_F)

// fp16, fp32, float4
#define INSTANTIATE_FP16_FP32_F4(DIM) INSTANTIATE(at::Half, float, float4, float, DIM)
FOR_EACH_DIM(INSTANTIATE_FP16_FP32_F4)

// fp16, fp32, float2
#define INSTANTIATE_FP16_FP32_F2(DIM) INSTANTIATE(at::Half, float, float2, float, DIM)
FOR_EACH_DIM(INSTANTIATE_FP16_FP32_F2)

// fp16, fp32, float
#define INSTANTIATE_FP16_FP32_F(DIM) INSTANTIATE(at::Half, float, float, float, DIM)
FOR_EACH_DIM(INSTANTIATE_FP16_FP32_F)

// bf16, bf16, float4
#define INSTANTIATE_BF16_BF16_F4(DIM) INSTANTIATE(at::BFloat16, at::BFloat16, float4, float, DIM)
FOR_EACH_DIM(INSTANTIATE_BF16_BF16_F4)

// bf16, bf16, float2
#define INSTANTIATE_BF16_BF16_F2(DIM) INSTANTIATE(at::BFloat16, at::BFloat16, float2, float, DIM)
FOR_EACH_DIM(INSTANTIATE_BF16_BF16_F2)

// bf16, bf16, float
#define INSTANTIATE_BF16_BF16_F(DIM) INSTANTIATE(at::BFloat16, at::BFloat16, float, float, DIM)
FOR_EACH_DIM(INSTANTIATE_BF16_BF16_F)

// bf16, fp32, float4
#define INSTANTIATE_BF16_FP32_F4(DIM) INSTANTIATE(at::BFloat16, float, float4, float, DIM)
FOR_EACH_DIM(INSTANTIATE_BF16_FP32_F4)

// bf16, fp32, float2
#define INSTANTIATE_BF16_FP32_F2(DIM) INSTANTIATE(at::BFloat16, float, float2, float, DIM)
FOR_EACH_DIM(INSTANTIATE_BF16_FP32_F2)

// bf16, fp32, float
#define INSTANTIATE_BF16_FP32_F(DIM) INSTANTIATE(at::BFloat16, float, float, float, DIM)
FOR_EACH_DIM(INSTANTIATE_BF16_FP32_F)

#undef INSTANTIATE
#undef FOR_EACH_DIM
