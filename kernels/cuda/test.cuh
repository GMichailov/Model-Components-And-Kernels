#pragma once

#include "norms/rmsnorm.cuh"
#include "norms/layernorm.cuh"
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>

// Test Correctness functions

namespace test {

__device__ __forceinline__ bool test_rms_norm_forward_divisible() {}

__device__ __forceinline__ bool test_rms_norm_forward_indivisible() {}
    
}