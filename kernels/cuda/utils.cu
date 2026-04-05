#include "utils.cuh"

namespace utils {

__global__ void warmup_kernel(float* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < 1024) data[idx] = data[idx] * 2.0f;
}

}
