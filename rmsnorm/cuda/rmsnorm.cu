#include <cuda_runtime.h>
#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
#include <iostream>

#define CHECK_CUDA_ERROR(val), check((val), #val, __FILE__, __LINE__)
void check (cudaError_t err, char const* func, char const* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA RUNTIME ERROR at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Issues: CUB load isn't as fast and potentially dangerous
template<typename ACTIVATION_DATA_TYPE, typename WEIGHT_DATA_TYPE, int DIM_X, int BLOCK_SIZE>
__global__ __forceinline__ void first_rmsnorm_kernel(
    const ACTIVATION_DATA_TYPE* __restrict__ x, const WEIGHT_DATA_TYPE* __restrict__ gamma, ACTIVATION_DATA_TYPE* __restrict__ y, int seq_len, float epsilon) 
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    // Token index is blockIdx.x
    const ACTIVATION_DATA_TYPE* x_row = x + blockIdx.x * DIM_X;
    ACTIVATION_DATA_TYPE* y_row = y + blockIdx.x * DIM_X;
    ACTIVATION_DATA_TYPE thread_sum = 0;
    #pragma unroll
    for (int i{threadIdx.x}; i < DIM_X; i += blockDim.x) {
        ACTIVATION_DATA_TYPE v = x_row[i];
        thread_sum += v*v;
    }
    float sum = BlockReduce(temp_storage).Sum(thread_sum);
    __shared__ ACTIVATION_DATA_TYPE inv_rms;
    if (threadIdx.x == 0) inv_rms = rsqrtf(sum / DIM_X + epsilon);
    __syncthreads();
    #pragma unroll
    for (int i{threadIdx.x}; i < DIM_X; i += blockDim.x) {
        y_row[i] = x_row[i] * inv_rms * gamma[i];
    }
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset{16}; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val) {
    static __shared__ float warp_sums[32];
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    val = warp_reduce_sum(val);
    if (lane == 0) warp_sums[warp] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / 32) ? warp_sums[lane] : 0.0f;
    if (warp == 0) val = warp_reduce_sum(val);
    return val;
}

// Assumes ACT DTYPE >= WEIGHT DTYPE (Literally no reason to handle the opposite)
// Currently assume DIM_X divisible by float 4.
// Cons: Only handles one token per block, want to make this handle more than one per block.
template<typename ACTIVATION_DATA_TYPE, typename WEIGHT_DATA_TYPE, int DIM_X, int BLOCK_SIZE>
__global__ __forceinline__ void second_rmsnorm_kernel(
    const ACTIVATION_DATA_TYPE* __restrict__ x, const WEIGHT_DATA_TYPE* __restrict__ gamma, ACTIVATION_DATA_TYPE* __restrict__ y, float epsilon) 
{
    constexpr int act_vec_loads = sizeof(float4) / sizeof(ACTIVATION_DATA_TYPE);
    ACTIVATION_DATA_TYPE x_vals[act_vec_loads];
    ACTIVATION_DATA_TYPE out[act_vec_loads];
    const ACTIVATION_DATA_TYPE* x_row = x + blockIdx.x * DIM_X;
    ACTIVATION_DATA_TYPE* y_row = y + blockIdx.x * DIM_X;
    float thread_sum = 0.0f;
    const float4* x4 = reinterpret_cast<const float4*>(x_row);
    #pragma unroll
    for (int i{threadIdx.x}; i < DIM_X/act_vec_loads; i += BLOCK_SIZE) {
        float4 v = x4[i];
        memcpy(x_vals, &v, sizeof(v));
        #pragma unroll
        for (int j{0}; j < act_vec_loads; ++j) {
            float val = float(x_vals[j]);
            thread_sum += v*v;
        }
    }
    float sum = block_reduce_sum(thread_sum);
    __shared__ float inv_rms;
    if (threadIdx.x == 0) inv_rms = rsqrtf(sum / DIM_X + epsilon);
    __syncthreads();
    float4* y4 = reinterpret_cast<float4*>(y_row);
    for (int i{threadIdx.x}; i < DIM_X/act_vec_loads; i += BLOCK_SIZE) {
        float4 x4v = x4[i];
        memcpy(x_vals, &x4v, sizeof(x4v));
        #pragma unroll
        for (int j{0}; j < act_vec_loads; ++j) {
            ACTIVATION_DATA_TYPE weight = ACTIVATION_DATA_TYPE([j + i * act_vec_loads]);
            out[j] = x_vals[j] * weight * inv_rms;
        }
        memcpy(y4 + i, out, sizeof(float4));
    }
}