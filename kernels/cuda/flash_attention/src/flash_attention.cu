#include "flash_attention.cuh"
#include <mma.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups.h>

namespace flash_attention {

namespace kernels {

    // Assumes QKV are [B, Hs, Seq, D_HEAD]
    // Assumes contiguity across DIM_X
    // Parallelized over num_heads and batch size and sequence length.
    // Optimizing this kernel for pretraining and SFT where all sequences are the same length.
    template<
        typename ACTIVATION_DTYPE, 
        int VECTORIZED_LOAD_COUNT, int DIM_HEAD, int THREADS_PER_BLOCK,
        int Q_TILE_ROWS, int KV_TILE_ROWS
    >
    __global__ void causal_flash_attention(
        const ACTIVATION_DTYPE* __restrict__ Q, const ACTIVATION_DTYPE* __restrict__ K, const ACTIVATION_DTYPE* __restrict__ V, ACTIVATION_DTYPE* __restrict__ O,
        const ACTIVATION_DTYPE* __restrict__ L,
        int target_batch, int sequence_chunks, int sub_seq_len, int curr_seq_chunk,
        int stride_batch_q, int stride_head_dim_q, int num_heads_q, int q_head,
        int stride_batch_kv, int stride_head_dim_kv, int num_heads_kv, int kv_head, // Denote which head this is.
    ) {
        // Need to store 2 x (2 x KV_TILE_ROWS x kv_head_dim) + Q_TILE_ROWS Logsum + Q_TILE_ROWS Max + Q_TILE_ROWS * KV_TILE_ROWS for O) for async buffer loading.
        extern __shared__ __align__(16) unsigned char raw_smem[];
        ACTIVATION_DTYPE* = reinterpret_cast<ACTIVATION_DTYPE*>(raw_smem);
        auto block = cooperative_groups::this_thread_block();
        __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> shared_pipe_state;
        auto pipe = cuda::make_pipeline(block, &shared_pipe_state);
        
        int buffer_iter{1};
        int smem_true_data{2 * KV_TILE_ROWS * DIM_HEAD + 2 * Q_TILE_ROWS + Q_TILE_ROWS * KV_TILE_ROWS};
        ACTIVATION_DTYPE* smem_K = smem;
        ACTIVATION_DTYPE* smem_V = smem + KV_TILE_ROWS * DIM_HEAD;
        ACTIVATION_DTYPE* smem_L = smem_V + KV_TILE_ROWS * DIM_HEAD;
        ACTIVATION_DTYPE* smem_max = smem_L + Q_TILE_ROWS;
        ACTIVATION_DTYPE* smem_O = smem_max + Q_TILE_ROWS;
        // Only loop is over head_dim until entire single respective heads have been compared.
        int kv_global_offset = target_batch * num_heads_kv * sequence_chunks * sub_seq_len * DIM_HEAD + curr_seq_chunk * sub_seq_len * DIM_HEAD;
        const ACTIVATION_DTYPE* gmem_k_row = K + offset;
        const ACTIVATION_DTYPE* gmem_v_row = V + offset;
        ACTIVATION_DTYPE* smem_curr_k = smem_K;
        ACTIVATION_DTYPE* smem_curr_v = smem_V;
        for (int kv_tile{0}; kv_tile < sub_seq_len; kv_tile += KV_TILE_ROWS) {
            gmem_k_row += kv_tile;
            gmem_v_row += kv_tile;
            smem_curr_k += buffer_iter * smem_true_data;
            smem_curr_v += buffer_iter * smem_true_data;
            pipe.producer_acquire();
            // TODO: OPTIMIZE THIS BULLSHIT LOOP
            #pragma unroll
            for (int kv_row{0}; kv_row < KV_TILE_ROWS; kv_row += 1) {
                cuda::memcpy_async(smem_curr_k, gmem_k_row, sizeof(ACTIVATION_DTYPE) * DIM_HEAD, pipe);
                cuda::memcpy_async(smem_curr_v, gmem_v_row, sizeof(ACTIVATION_DTYPE) * DIM_HEAD, pipe);
            }
            pipe.producer_commit();
            // Init l and m in smem.
            // Pull Q into registers
        }
        buffer_iter = -buffer_iter;
    }

}

}