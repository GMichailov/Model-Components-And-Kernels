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

    template<typename scalar_t, int vec_size>
    struct alignas(sizeof(scalar_t) * vec_size) aligned_vector{ scalar_t val[vec_size]; };

    template<typename wgt_scalar_t, int vec_size>
    struct alignas(sizeof(wgt_scalar_t) * vec_size) aligned_wgt_vector{ wgt_scalar_t val[vec_size]; };

    // Currently enforces DIM_X divisible by VECTORIZED_LOAD_TYPE.
    template<typename ACTIVATION_DTYPE, typename WEIGHT_DTYPE, typename CALCULATION_DTYPE, int VECTORIZED_LOAD_COUNT, int DIM_X, int THREADS_PER_BLOCK>
    __global__ void rmsnorm_kernel_divisible_forward(
        const ACTIVATION_DTYPE* __restrict__ x, const WEIGHT_DTYPE* __restrict__ gamma, ACTIVATION_DTYPE* __restrict__ y, float epsilon)
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
        const wgt_vec_t* gamma_vec = reinterpret_cast<const wgt_vec_t*>(gamma);
        #pragma unroll
        for (int i{threadIdx.x}; i < num_loads; i += THREADS_PER_BLOCK) {
            vec_t v = x_vec[i];
            #pragma unroll
            for (int j{0}; j < VECTORIZED_LOAD_COUNT; ++j) {
                CALCULATION_DTYPE val = static_cast<CALCULATION_DTYPE>(v.val[j]);
                thread_sum += val*val;
            }
        }
        CALCULATION_DTYPE sum = block_reduce_sum<CALCULATION_DTYPE>(thread_sum);
        __shared__ CALCULATION_DTYPE inv_rms;
        if (threadIdx.x == 0) inv_rms = static_cast<CALCULATION_DTYPE>(rsqrtf(sum / DIM_X + epsilon));
        __syncthreads();
        vec_t* y_vec = reinterpret_cast<vec_t*>(y_row);
        for (uint i{threadIdx.x}; i < num_loads; i += THREADS_PER_BLOCK) {
            vec_t x_vec_val = x_vec[i];
            wgt_vec_t gamma_vec_val = gamma_vec[i];
            vec_t out;
            #pragma unroll
            for (int j{0}; j < VECTORIZED_LOAD_COUNT; ++j) {
                out.val[j] = static_cast<ACTIVATION_DTYPE>(static_cast<CALCULATION_DTYPE>(x_vec_val.val[j]) * static_cast<CALCULATION_DTYPE>(gamma_vec_val.val[j]) * inv_rms);
            }
            y_vec[i] = out;
        }
    }

    // Currently allows DIM_X to be indivisible by VECTORIZED_LOAD_TYPE.
    template<typename ACTIVATION_DTYPE, typename WEIGHT_DTYPE, typename CALCULATION_DTYPE, int VECTORIZED_LOAD_COUNT, int DIM_X, int THREADS_PER_BLOCK>
    __global__ void rmsnorm_kernel_indivisible_forward(
        const ACTIVATION_DTYPE* __restrict__ x, const WEIGHT_DTYPE* __restrict__ gamma, ACTIVATION_DTYPE* __restrict__ y, float epsilon)
    {
        using vec_t = aligned_vector<ACTIVATION_DTYPE, VECTORIZED_LOAD_COUNT>;
        static_assert(THREADS_PER_BLOCK <= 1024, "Max threads per block is 1024.");
        static_assert(sizeof(vec_t) % sizeof(ACTIVATION_DTYPE) == 0, "Vectorized load type must be a a multiple of the activation size.");
        static_assert(sizeof(vec_t) >= sizeof(ACTIVATION_DTYPE), "Vectorized load type must be the same or larger than the activation type.");
        static_assert(sizeof(CALCULATION_DTYPE) >= sizeof(ACTIVATION_DTYPE), "Calculation accumulation datatype must be a larger size than activation type.");
        constexpr int act_vec_loads = sizeof(vec_t) / sizeof(ACTIVATION_DTYPE);
        constexpr int num_loads = DIM_X / act_vec_loads;
        constexpr int tail = DIM_X % act_vec_loads;
        int tail_idx = num_loads * act_vec_loads + threadIdx.x;
        ACTIVATION_DTYPE x_vals[act_vec_loads];
        ACTIVATION_DTYPE out[act_vec_loads];
        const ACTIVATION_DTYPE* x_row = x + blockIdx.x * DIM_X;
        ACTIVATION_DTYPE* y_row = y + blockIdx.x * DIM_X;
        CALCULATION_DTYPE thread_sum{0};
        const vec_t* x_vec = reinterpret_cast<const vec_t*>(x_row);
        #pragma unroll
        for (uint i{threadIdx.x}; i < num_loads; i += THREADS_PER_BLOCK) {
            vec_t v = x_vec[i];
            memcpy(x_vals, &v, sizeof(v));
            #pragma unroll
            for (int j{0}; j < act_vec_loads; ++j) {
                CALCULATION_DTYPE val = static_cast<CALCULATION_DTYPE>(x_vals[j]);
                thread_sum += val * val;
            }
        }
        if(threadIdx.x < tail) { CALCULATION_DTYPE val = static_cast<CALCULATION_DTYPE>(x_row[tail_idx]); thread_sum += val * val; }
        CALCULATION_DTYPE sum = block_reduce_sum<CALCULATION_DTYPE>(thread_sum);
        __shared__ CALCULATION_DTYPE inv_rms;
        if (threadIdx.x == 0) inv_rms = static_cast<CALCULATION_DTYPE>(rsqrtf(sum / DIM_X + epsilon));
        __syncthreads();
        vec_t* y_vec = reinterpret_cast<vec_t*>(y_row);
        for (uint i{threadIdx.x}; i < num_loads; i += THREADS_PER_BLOCK) {
            vec_t x_vec_val = x_vec[i];
            memcpy(x_vals, &x_vec_val, sizeof(x_vec_val));
            #pragma unroll
            for (int j{0}; j < act_vec_loads; ++j) {
                CALCULATION_DTYPE weight = static_cast<CALCULATION_DTYPE>(gamma[j + i * act_vec_loads]);
                out[j] = static_cast<ACTIVATION_DTYPE>(static_cast<CALCULATION_DTYPE>(x_vals[j]) * weight * inv_rms);
            }
            vec_t out_vec;
            memcpy(&out_vec, out, sizeof(out));
            y_vec[i] = out_vec;
        }
        if (threadIdx.x < tail) { y_row[tail_idx] = static_cast<ACTIVATION_DTYPE>(static_cast<CALCULATION_DTYPE>(x_row[tail_idx]) * static_cast<CALCULATION_DTYPE>(gamma[tail_idx]) * inv_rms); }
    }
}

namespace launchers {

    namespace offline {

        template<typename ACTIVATION_DTYPE, typename WEIGHT_DTYPE, typename CALCULATION_DTYPE, int VECTORIZED_LOAD_COUNT, int DIM_X, int THREADS_PER_BLOCK>
        at::Tensor cuda_rmsnorm_divisible_forward_launcher_offline(const at::Tensor &x, const at::Tensor &gamma, float epsilon) {
            using vec_t = kernels::aligned_vector<ACTIVATION_DTYPE, VECTORIZED_LOAD_COUNT>;
            TORCH_CHECK(x.dim() == 2, "x must be 2D, got dim ", x.dim());
            TORCH_CHECK(x.size(1) == DIM_X, "Expected hidden dim ", DIM_X, " but got ", x.size(1));
            TORCH_CHECK(gamma.numel() == DIM_X, "Expected gamma len ", DIM_X, " but got ", gamma.numel());
            TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr<ACTIVATION_DTYPE>()) % alignof(vec_t) == 0, "x is not properly aligned for vectorized access");
            at::Tensor y = at::empty_like(x);
            TORCH_CHECK(reinterpret_cast<uintptr_t>(y.data_ptr<ACTIVATION_DTYPE>()) % alignof(vec_t) == 0, "y is not properly aligned for vectorized access");
            dim3 grid(x.numel() / DIM_X);
            dim3 block(THREADS_PER_BLOCK);
            kernels::rmsnorm_kernel_divisible_forward<ACTIVATION_DTYPE, WEIGHT_DTYPE, CALCULATION_DTYPE, VECTORIZED_LOAD_COUNT, DIM_X, THREADS_PER_BLOCK><<<grid, block>>>(x.data_ptr<ACTIVATION_DTYPE>(), gamma.data_ptr<WEIGHT_DTYPE>(), y.data_ptr<ACTIVATION_DTYPE>(), epsilon);
            return y;
        }

        template<typename ACTIVATION_DTYPE, typename WEIGHT_DTYPE, typename CALCULATION_DTYPE, int VECTORIZED_LOAD_COUNT, int DIM_X, int THREADS_PER_BLOCK>
        at::Tensor cuda_rmsnorm_indivisible_forward_launcher_offline(const at::Tensor &x, const at::Tensor &gamma, float epsilon) {
            using vec_t = kernels::aligned_vector<ACTIVATION_DTYPE, VECTORIZED_LOAD_COUNT>;
            TORCH_CHECK(x.dim() == 2, "x must be 2D, got dim ", x.dim());
            TORCH_CHECK(x.size(1) == DIM_X, "Expected hidden dim ", DIM_X, " but got ", x.size(1));
            TORCH_CHECK(gamma.numel() == DIM_X, "Expected gamma len ", DIM_X, " but got ", gamma.numel());
            TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr<ACTIVATION_DTYPE>()) % alignof(vec_t) == 0, "x is not properly aligned for vectorized access");
            at::Tensor y = at::empty_like(x);
            TORCH_CHECK(reinterpret_cast<uintptr_t>(y.data_ptr<ACTIVATION_DTYPE>()) % alignof(vec_t) == 0, "y is not properly aligned for vectorized access");
            dim3 grid(x.numel() / DIM_X);
            dim3 block(THREADS_PER_BLOCK);
            kernels::rmsnorm_kernel_indivisible_forward<ACTIVATION_DTYPE, WEIGHT_DTYPE, CALCULATION_DTYPE, VECTORIZED_LOAD_COUNT, DIM_X, THREADS_PER_BLOCK><<<grid, block>>>(x.data_ptr<ACTIVATION_DTYPE>(), gamma.data_ptr<WEIGHT_DTYPE>(), y.data_ptr<ACTIVATION_DTYPE>(), epsilon);
            return y;
        }
    }

    namespace online {



    }

}

}

// ============================================================================
// Explicit template instantiations
// Dimensions: 128, 256, 512
// Threads: 128, 256, 512
// ============================================================================

// FP32 (ACT=float, WGT=float, CALC=float) - VLCOUNT=1,2,4
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 1, 128, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 1, 128, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 1, 128, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 1, 256, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 1, 256, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 1, 256, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 1, 512, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 1, 512, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 1, 512, 512>(const at::Tensor&, const at::Tensor&, float);

template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 2, 128, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 2, 128, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 2, 128, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 2, 256, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 2, 256, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 2, 256, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 2, 512, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 2, 512, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 2, 512, 512>(const at::Tensor&, const at::Tensor&, float);

template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 4, 128, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 4, 128, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 4, 128, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 4, 256, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 4, 256, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 4, 256, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 4, 512, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 4, 512, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<float, float, float, 4, 512, 512>(const at::Tensor&, const at::Tensor&, float);

// FP16/FP16/FP32 (ACT=at::Half, WGT=at::Half, CALC=float) - VLCOUNT=1,2,4,8
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 1, 128, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 1, 128, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 1, 128, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 1, 256, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 1, 256, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 1, 256, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 1, 512, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 1, 512, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 1, 512, 512>(const at::Tensor&, const at::Tensor&, float);

template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 2, 128, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 2, 128, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 2, 128, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 2, 256, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 2, 256, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 2, 256, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 2, 512, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 2, 512, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 2, 512, 512>(const at::Tensor&, const at::Tensor&, float);

template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 4, 128, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 4, 128, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 4, 128, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 4, 256, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 4, 256, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 4, 256, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 4, 512, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 4, 512, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 4, 512, 512>(const at::Tensor&, const at::Tensor&, float);

template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 8, 128, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 8, 128, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 8, 128, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 8, 256, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 8, 256, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 8, 256, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 8, 512, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 8, 512, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, at::Half, float, 8, 512, 512>(const at::Tensor&, const at::Tensor&, float);

// FP16/FP32/FP32 (ACT=at::Half, WGT=float, CALC=float) - VLCOUNT=1,2,4,8
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 1, 128, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 1, 128, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 1, 128, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 1, 256, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 1, 256, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 1, 256, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 1, 512, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 1, 512, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 1, 512, 512>(const at::Tensor&, const at::Tensor&, float);

template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 2, 128, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 2, 128, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 2, 128, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 2, 256, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 2, 256, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 2, 256, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 2, 512, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 2, 512, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 2, 512, 512>(const at::Tensor&, const at::Tensor&, float);

template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 4, 128, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 4, 128, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 4, 128, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 4, 256, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 4, 256, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 4, 256, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 4, 512, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 4, 512, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 4, 512, 512>(const at::Tensor&, const at::Tensor&, float);

template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 8, 128, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 8, 128, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 8, 128, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 8, 256, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 8, 256, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 8, 256, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 8, 512, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 8, 512, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::Half, float, float, 8, 512, 512>(const at::Tensor&, const at::Tensor&, float);

// BF16/BF16/FP32 (ACT=at::BFloat16, WGT=at::BFloat16, CALC=float) - VLCOUNT=1,2,4,8
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 128, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 128, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 128, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 256, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 256, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 256, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 512, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 512, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 512, 512>(const at::Tensor&, const at::Tensor&, float);

template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 128, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 128, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 128, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 256, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 256, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 256, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 512, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 512, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 512, 512>(const at::Tensor&, const at::Tensor&, float);

template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 128, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 128, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 128, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 256, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 256, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 256, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 512, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 512, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 512, 512>(const at::Tensor&, const at::Tensor&, float);

template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 128, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 128, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 128, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 256, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 256, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 256, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 512, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 512, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 512, 512>(const at::Tensor&, const at::Tensor&, float);

// BF16/FP32/FP32 (ACT=at::BFloat16, WGT=float, CALC=float) - VLCOUNT=1,2,4,8
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 1, 128, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 1, 128, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 1, 128, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 1, 256, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 1, 256, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 1, 256, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 1, 512, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 1, 512, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 1, 512, 512>(const at::Tensor&, const at::Tensor&, float);

template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 2, 128, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 2, 128, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 2, 128, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 2, 256, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 2, 256, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 2, 256, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 2, 512, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 2, 512, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 2, 512, 512>(const at::Tensor&, const at::Tensor&, float);

template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 4, 128, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 4, 128, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 4, 128, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 4, 256, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 4, 256, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 4, 256, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 4, 512, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 4, 512, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 4, 512, 512>(const at::Tensor&, const at::Tensor&, float);

template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 8, 128, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 8, 128, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 8, 128, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 8, 256, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 8, 256, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 8, 256, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 8, 512, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 8, 512, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<at::BFloat16, float, float, 8, 512, 512>(const at::Tensor&, const at::Tensor&, float);

// ============================================================================
// Indivisible forward launcher template instantiations
// ============================================================================

// FP32 (ACT=float, WGT=float, CALC=float) - VLCOUNT=1,2,4
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 1, 128, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 1, 128, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 1, 128, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 1, 256, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 1, 256, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 1, 256, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 1, 512, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 1, 512, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 1, 512, 512>(const at::Tensor&, const at::Tensor&, float);

template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 2, 128, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 2, 128, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 2, 128, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 2, 256, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 2, 256, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 2, 256, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 2, 512, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 2, 512, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 2, 512, 512>(const at::Tensor&, const at::Tensor&, float);

template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 4, 128, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 4, 128, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 4, 128, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 4, 256, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 4, 256, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 4, 256, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 4, 512, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 4, 512, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<float, float, float, 4, 512, 512>(const at::Tensor&, const at::Tensor&, float);

// FP16/FP16/FP32 (ACT=at::Half, WGT=at::Half, CALC=float) - VLCOUNT=1,2,4,8
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 1, 128, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 1, 128, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 1, 128, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 1, 256, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 1, 256, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 1, 256, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 1, 512, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 1, 512, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 1, 512, 512>(const at::Tensor&, const at::Tensor&, float);

template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 2, 128, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 2, 128, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 2, 128, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 2, 256, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 2, 256, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 2, 256, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 2, 512, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 2, 512, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 2, 512, 512>(const at::Tensor&, const at::Tensor&, float);

template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 4, 128, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 4, 128, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 4, 128, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 4, 256, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 4, 256, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 4, 256, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 4, 512, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 4, 512, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 4, 512, 512>(const at::Tensor&, const at::Tensor&, float);

template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 8, 128, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 8, 128, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 8, 128, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 8, 256, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 8, 256, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 8, 256, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 8, 512, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 8, 512, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 8, 512, 512>(const at::Tensor&, const at::Tensor&, float);

// FP16/FP32/FP32 (ACT=at::Half, WGT=float, CALC=float) - VLCOUNT=1,2,4,8
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 1, 128, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 1, 128, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 1, 128, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 1, 256, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 1, 256, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 1, 256, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 1, 512, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 1, 512, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 1, 512, 512>(const at::Tensor&, const at::Tensor&, float);

template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 2, 128, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 2, 128, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 2, 128, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 2, 256, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 2, 256, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 2, 256, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 2, 512, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 2, 512, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 2, 512, 512>(const at::Tensor&, const at::Tensor&, float);

template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 4, 128, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 4, 128, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 4, 128, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 4, 256, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 4, 256, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 4, 256, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 4, 512, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 4, 512, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 4, 512, 512>(const at::Tensor&, const at::Tensor&, float);

template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 8, 128, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 8, 128, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 8, 128, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 8, 256, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 8, 256, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 8, 256, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 8, 512, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 8, 512, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, float, float, 8, 512, 512>(const at::Tensor&, const at::Tensor&, float);

// BF16/BF16/FP32 (ACT=at::BFloat16, WGT=at::BFloat16, CALC=float) - VLCOUNT=1,2,4,8
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 128, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 128, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 128, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 256, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 256, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 256, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 512, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 512, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 1, 512, 512>(const at::Tensor&, const at::Tensor&, float);

template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 128, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 128, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 128, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 256, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 256, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 256, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 512, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 512, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 2, 512, 512>(const at::Tensor&, const at::Tensor&, float);

template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 128, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 128, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 128, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 256, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 256, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 256, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 512, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 512, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 4, 512, 512>(const at::Tensor&, const at::Tensor&, float);

template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 128, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 128, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 128, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 256, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 256, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 256, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 512, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 512, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, at::BFloat16, float, 8, 512, 512>(const at::Tensor&, const at::Tensor&, float);

// BF16/FP32/FP32 (ACT=at::BFloat16, WGT=float, CALC=float) - VLCOUNT=1,2,4,8
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 1, 128, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 1, 128, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 1, 128, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 1, 256, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 1, 256, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 1, 256, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 1, 512, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 1, 512, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 1, 512, 512>(const at::Tensor&, const at::Tensor&, float);

template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 2, 128, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 2, 128, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 2, 128, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 2, 256, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 2, 256, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 2, 256, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 2, 512, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 2, 512, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 2, 512, 512>(const at::Tensor&, const at::Tensor&, float);

template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 4, 128, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 4, 128, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 4, 128, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 4, 256, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 4, 256, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 4, 256, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 4, 512, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 4, 512, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 4, 512, 512>(const at::Tensor&, const at::Tensor&, float);

template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 8, 128, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 8, 128, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 8, 128, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 8, 256, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 8, 256, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 8, 256, 512>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 8, 512, 128>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 8, 512, 256>(const at::Tensor&, const at::Tensor&, float);
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::BFloat16, float, float, 8, 512, 512>(const at::Tensor&, const at::Tensor&, float);

// ============================================================================
// Example: Truly indivisible dimension (130 % 8 = 2, tail of 2 elements)
// ============================================================================
template at::Tensor norms::launchers::offline::cuda_rmsnorm_indivisible_forward_launcher_offline<at::Half, at::Half, float, 8, 130, 256>(const at::Tensor&, const at::Tensor&, float);
