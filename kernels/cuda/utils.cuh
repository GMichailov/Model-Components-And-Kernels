#pragma once

#include <torch/torch.h>
#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>

namespace utils {

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

}

namespace dtypes {

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

    template<typename scalar_t, int vec_size>
    struct alignas(sizeof(scalar_t) * vec_size) aligned_vector{ scalar_t val[vec_size]; };

}

}