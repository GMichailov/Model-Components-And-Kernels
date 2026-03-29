# pragma once

#include "../norms/rmsnorm.cuh"

#include <random>
#include <stdexcept>
#include <vector>
#include <cassert>

// Create rand vector of vector of vectors. Assume [B*S, D] for simplicity.
template<typename T>
const T* create_rand_arr(size_t batch_and_seq, size_t dim_x) {}

// Number of blocks spawned will always be batch_and_seq (one block per row).
// Varying values will be dim_x and number of threads.
template<typename ACTIVATION_DATA_TYPE, typename WEIGHT_DATA_TYPE>
bool compare_same_dim_x(size_t batch_and_seq, size_t dim_x, std::vector<int> num_threads) {
    const ACTIVATION_DATA_TYPE* x{create_rand_arr<ACTIVATION_DATA_TYPE>(batch_and_seq, dim_x)};
    const WEIGHT_DATA_TYPE* gamma{create_rand_arr<WEIGHT_DATA_TYPE>(1, dim_x)};
    const ACTIVATION_DATA_TYPE* d_x;
    ACTIVATION_DATA_TYPE* d_y;
    const WEIGHT_DATA_TYPE* d_gamma;

    CHECK_CUDA_ERROR(cudaMallocAsync(&d_x, sizeof(ACTIVATION_DATA_TYPE) * batch_and_seq * dim_x));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_y, sizeof(ACTIVATION_DATA_TYPE) * batch_and_seq * dim_x));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_gamma, sizeof(WEIGHT_DATA_TYPE) * dim_x));
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_x, x, sizeof(ACTIVATION_DATA_TYPE) * batch_and_seq * dim_x, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_y, y, sizeof(ACTIVATION_DATA_TYPE) * batch_and_seq * dim_x, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_gamma, gamma, sizeof(ACTIVATION_DATA_TYPE) * dim_x, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();

    utils::gpu_warmup();

    // Execute and benchmark.

    for (int threads : num_threads) {
        
    }

    cudaDeviceSynchronize();
    // Error check.
    cudaError_t last_error{cudaGetLastError()};
    if (last_error != cudaSuccess) {
        std::cerr << "Failed to execute." << std::endl;
        std::cerr << cudaGetErrorString(last_error) << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Free memory.
    CHECK_CUDA_ERROR(cudaFree(d_x));
    CHECK_CUDA_ERROR(cudaFree(d_y));
    CHECK_CUDA_ERROR(cudaFree(d_gamma));
    delete x; delete y; delete gamma;
    std::exit(EXIT_SUCCESS); 
}
