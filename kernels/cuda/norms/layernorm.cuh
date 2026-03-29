#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA_ERROR(val), check((val), #val, __FILE__, __LINE__)
void check (cudaError_t err, char const* func, char const* file, int line) {
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

// Currently assumes that bits evenly divide into dim x when doing vectorized loads.
template<typename ACTIVATION_DTYPE, typename WEIGHT_DTYPE, typename VECTORIZED_LOAD_TYPE, int DIM_X>
__global__ __forceinline__ void layer_norm_no_bias_forward(
    const ACTIVATION_DTYPE* __restrict__ x, const WEIGHT_DTYPE* __restrict__ gamma, ACTIVATION_DTYPE* __restrict__ y, float epsilon
) {
    __shared__ float average;
    __shared__ float denom;
    constexpr int act_vec_loads{sizeof(VECTORIZED_LOAD_TYPE) / sizeof(ACTIVATION_DTYPE)};
    constexpr int num_loads{DIM_X / sizeof(VECTORIZED_LOAD_TYPE)};
    ACTIVATION_DTYPE buf_x[act_vec_loads];
    ACTIVATION_DTYPE out[act_vec_loads];
    const ACTIVATION_DTYPE* row_x = x + blockIdx.x * DIM_X;
    float thread_sum = 0.0f;
    const VECTORIZED_LOAD_TYPE* vec_x = reinterpret_cast<const ACTIVATION_DTYPE*>(row_x);
    // Calculate Mean.
    #pragma unroll
    for (int i{threadIdx.x}; i < num_loads; i += blockDim.x) {
        VECTORIZED_LOAD_TYPE v = vec_x[i];
        memcpy(buf_x, &v, sizeof(VECTORIZED_LOAD_TYPE));
        #pragma unroll
        for (int j{0}; j < act_vec_loads; ++j) {
            float val = float(buf_x[j]);
            thread_sum += val * val;
        }
    }
    float sum = block_reduce_sum(thread_sum);
    if (threadIdx.x == 0) average = sum / DIM_X;
    __syncthreads();
    // Calculate variance.
    thread_sum = 0.0f;
    #pragma unroll
    for (int i{threadIdx.x}; i < num_loads; i += blockDim.x) {
        VECTORIZED_LOAD_TYPE v = vec_x[i];
        memcpy(buf_x, &v, sizeof(VECTORIZED_LOAD_TYPE));
        #pragma unroll
        for (int j{0}; j < act_vec_loads; ++j) {
            float val = float(buf_x[j]);
            float stdv{val * val - average};
            thread_sum += sqrt(stdv * stdv + epsilon);
        }
    }
    sum = block_reduce_sum(thread_sum);
    if (threadIdx.x == 0) denom = sum;
    __syncthreads();
    // Calculate output.
    y = y[blockIdx.x * DIM_X];
    VECTORIZED_LOAD_TYPE* vec_y = reinterpret_cast<ACTIVATION_DTYPE>(y);
    for (int i{threadIdx.x}; i < act_vec_loads; ++i) {
        VECTORIZED_LOAD_TYPE vec_x_vals = vec_x[i];
        memcpy(buf_x, &vec_x_vals, sizeof(ACTIVATION_DTYPE));
        #pragma unroll
        for (int j{0}; j < act_vec_loads; ++j) {
            ACTIVATION_DTYPE weight = ACTIVATION_DTYPE(gamma[j + i * act_vec_loads]);
            out[j] = gamma * (buf_x[j] - average) / denom; 
        }
        memcpy(vec_y + i, out, sizeof(VECTORIZED_LOAD_TYPE));
    }
}
