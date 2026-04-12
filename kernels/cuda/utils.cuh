#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <torch/torch.h>

namespace utils {
/*
#define CHECK_CUDA_ERROR(val), check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, char const* func, char const* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA RUNTIME ERROR at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
*/
void gpu_warmup();

__global__ void warmup_kernel(float* data);

struct optimalNormLaunchConfig {
    int num_threads;
    int num_blocks;
    at::ScalarType vectorized_load_type;
    int hidden_size;
};

}
