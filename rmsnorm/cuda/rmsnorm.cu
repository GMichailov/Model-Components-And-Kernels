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

/*
template<typename ACTIVATION_DATA_TYPE, typename WEIGHT_DATA_TYPE, int DIM_X, int BLOCK_SIZE>
__global__ __forceinline__ void rmsnorm_cub_kernel(
    const ACTIVATION_DATA_TYPE* __restrict__ x, const WEIGHT_DATA_TYPE* __restrict__ gamma, ACTIVATION_DATA_TYPE* __restrict__ y, float epsilon) 
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    // Token index is blockIdx.x
    const ACTIVATION_DATA_TYPE* x_row = x + blockIdx.x * DIM_X;
    ACTIVATION_DATA_TYPE* y_row = y + blockIdx.x * DIM_X;
    ACTIVATION_DATA_TYPE thread_sum = 0.0f;
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
*/

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

// Assumes size ACT DTYPE >= WEIGHT DTYPE (Literally no reason to handle the opposite)
// Currently enforces DIM_X divisible by VECTORIZED_LOAD_TYPE.
template<typename ACTIVATION_DATA_TYPE, typename WEIGHT_DATA_TYPE, typename VECTORIZED_LOAD_TYPE, int DIM_X>
__global__ __forceinline__ void noncub_rmsnorm_kernel_divisible(
    const ACTIVATION_DATA_TYPE* __restrict__ x, const WEIGHT_DATA_TYPE* __restrict__ gamma, ACTIVATION_DATA_TYPE* __restrict__ y, float epsilon) 
{
    static_assert(DIM_X % VECTORIZED_LOAD_TYPE == 0, "Vectorized load datatype must be a divisor of dim_x.");
    constexpr int act_vec_loads = sizeof(VECTORIZED_LOAD_TYPE) / sizeof(ACTIVATION_DATA_TYPE);
    constexpr int num_loads = DIM_X / sizeof(VECTORIZED_LOAD_TYPE);
    ACTIVATION_DATA_TYPE x_vals[act_vec_loads];
    ACTIVATION_DATA_TYPE out[act_vec_loads];
    const ACTIVATION_DATA_TYPE* x_row = x + blockIdx.x * DIM_X;
    ACTIVATION_DATA_TYPE* y_row = y + blockIdx.x * DIM_X;
    float thread_sum = 0.0f;
    const VECTORIZED_LOAD_TYPE* x_vec = reinterpret_cast<const VECTORIZED_LOAD_TYPE*>(x_row);
    #pragma unroll
    for (int i{threadIdx.x}; i < num_loads; i += blockDim.x) {
        VECTORIZED_LOAD_TYPE v = x_vec[i];
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
    VECTORIZED_LOAD_TYPE* y_vec = reinterpret_cast<VECTORIZED_LOAD_TYPE*>(y_row);
    for (int i{threadIdx.x}; i < num_loads; i += blockDim.x) {
        VECTORIZED_LOAD_TYPE x_vec_val = x_vec[i];
        memcpy(x_vals, &x_vec_val, sizeof(x_vec_val));
        #pragma unroll
        for (int j{0}; j < act_vec_loads; ++j) {
            ACTIVATION_DATA_TYPE weight = ACTIVATION_DATA_TYPE(gamma[j + i * act_vec_loads]);
            out[j] = x_vals[j] * weight * inv_rms;
        }
        memcpy(y_vec + i, out, sizeof(VECTORIZED_LOAD_TYPE));
    }
}

// Assumes size ACT DTYPE >= WEIGHT DTYPE (Literally no reason to handle the opposite)
// Currently does NOT assume DIM_X divisible by VECTORIZED_LOAD_TYPE. (Can handle both scenarios though).
template<typename ACTIVATION_DATA_TYPE, typename WEIGHT_DATA_TYPE, typename VECTORIZED_LOAD_TYPE, int DIM_X>
__global__ __forceinline__ void noncub_rmsnorm_kernel_indivisible(
    const ACTIVATION_DATA_TYPE* __restrict__ x, const WEIGHT_DATA_TYPE* __restrict__ gamma, ACTIVATION_DATA_TYPE* __restrict__ y, float epsilon) 
{
    constexpr int act_vec_loads = sizeof(VECTORIZED_LOAD_TYPE) / sizeof(ACTIVATION_DATA_TYPE);
    constexpr int num_loads = DIM_X / act_vec_loads;
    constexpr int tail = DIM_X % act_vec_loads;
    int tail_idx = num_loads * act_vec_loads + threadIdx.x;
    ACTIVATION_DATA_TYPE x_vals[act_vec_loads];
    ACTIVATION_DATA_TYPE out[act_vec_loads];
    const ACTIVATION_DATA_TYPE* x_row = x + blockIdx.x * DIM_X;
    ACTIVATION_DATA_TYPE* y_row = y + blockIdx.x * DIM_X;
    float thread_sum{0.0f};
    const VECTORIZED_LOAD_TYPE* x_vec = reinterpret_cast<const VECTORIZED_LOAD_TYPE*>(x_row);
    #pragma unroll
    for (int i{threadIdx.x}; i < num_loads; i += blockDim.x) {
        VECTORIZED_LOAD_TYPE v = x_vec[i];
        memcpy(x_vals, &v, sizeof(v));
        #pragma unroll
        for (int j{0}; j < act_vec_loads; ++j) {
            float val = float(x_vals[j]);
            thread_sum += val * val;
        }
    }
    if (threadIdx.x < tail) thread_sum += float(x_row[tail_idx]);
    float sum = block_reduce_sum(thread_sum);
    __shared__ float inv_rms;
    if (threadIdx.x == 0) inv_rms = rsqrtf(sum / DIM_X + epsilon);
    __syncthreads();
    VECTORIZED_LOAD_TYPE* y_vec = reinterpret_cast<VECTORIZED_LOAD_TYPE*>(y_row);
    for (int i{threadIdx.x}; i < num_loads; i += blockDim.x) {
        VECTORIZED_LOAD_TYPE x_vec_val = x_vec[i];
        memcpy(x_vals, &x_vec_val, sizeof(x_vec_val));
        int gamma_idx = i * act_vec_loads;
        #pragma unroll
        for (int j{0}; j < act_vec_loads; ++j) {
            ACTIVATION_DATA_TYPE weight = ACTIVATION_DATA_TYPE(gamma[j + gamma_idx]);
            out[j] = x_vals[j] * weight * inv_rms;
        }
        memcpy(y_vec + i, out, sizeof(VECTORIZED_LOAD_TYPE));
    }
    if (threadIdx.x < tail) y_row[tail_idx] = x_row[tail_idx] * ACTIVATION_DATA_TYPE(gamma[tail_idx]) * inv_rms;
}
