#pragma once

#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>

namespace flash_attention {

namespace kernels {

    template<
        typename ACTIVATION_DTYPE, 
        int VECTORIZED_LOAD_COUNT, int DIM_X, int THREADS_PER_BLOCK, 
        int Q_TILE_ROWS, int KV_TILE_ROWS,
    >
    __global__ void flash_attention(
        const ACTIVATION_DTYPE* __restrict__ Q, const ACTIVATION_DTYPE* __restrict__ K, const ACTIVATION_DTYPE* __restrict__ V,
        int seq_len, 
        int stride_batch_q, int stride_head_dim_q, int num_heads_q,
        int stride_batch_kv, int stride_head_dim_kv, int num_heads_kv
    );

}

}