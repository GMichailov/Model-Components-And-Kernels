#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <torch/torch.h>

namespace utils {

#define CHECK_CUDA_ERROR(val), check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, char const* func, char const* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA RUNTIME ERROR at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

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

// Warm up
__global__ void warmup_kernel(float* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < 1024) data[idx] = data[idx] * 2.0f;
}

void gpu_warmup() {
    float* d_data;
    cudaMalloc(&d_data, 1024*sizeof(float));
    cudaMemset(d_data, 1, 1024*sizeof(float));
    warmup_kernel<<<8, 256>>>(d_data);
    cudaDeviceSynchronize();
    cudaFree(d_data);
}

// Optimal Configs

struct optimalNormLaunchConfig {
    int num_threads;
    int num_blocks;
    // at::ScalarType activation_dtype;
    // at::ScalarType weight_dtype;
    at::ScalarType vectorized_load_type;
    int hidden_size;
};

}