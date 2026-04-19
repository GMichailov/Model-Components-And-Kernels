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
    // Parallelized over num_heads and batch size and sequence length by each block handling one head only [1, 1, Seq, D_HEAD]
    // and then further parallelize By only doing calculations on a single Q tile per block.
    // Block dim3 dispatch: x=batches, y=q_heads, z=q_tiles
    // Warp dim3 dispatch: x=warp_id, y = row responsible for, z=thread idx within row (for optimal matrix matmul on tensor cores) (lowkey 16x16 subtiles) so y is 0-15 
    // When going for gqa, make the launcher flatten the tensor to open up more room.
    template<
        typename ACTIVATION_DTYPE, 
        int VECTORIZED_LOAD_COUNT, int DIM_HEAD, int NUM_WARPS,
        int Q_TILE_ROWS, int KV_TILE_ROWS
    >
    __global__ void causal_mha_flash_attention_training(
        const ACTIVATION_DTYPE* __restrict__ Q, const ACTIVATION_DTYPE* __restrict__ K, const ACTIVATION_DTYPE* __restrict__ V, ACTIVATION_DTYPE* __restrict__ O,
        ACTIVATION_DTYPE* __restrict__ L,
        int seq_len,
        int stride_batch_q, int stride_head_dim_q, int stride_batch_kv, int stride_head_dim_kv
    ) {
        // Assume tensor core math done 16x16x16 for now
        // Add static asserts later and be careful of overlap when holding a matrix.
        constexpr int q_load_size = Q_TILE_ROWS * DIM_HEAD / (NUM_WARPS * 32);
        constexpr int kv_load_size = KV_TILE_ROWS * DIM_HEAD / (NUM_WARPS * 32);
        constexpr int hold_size = 256 / (NUM_WARPS * 32);
        ACTIVATION_DTYPE frag_q[hold_size];
        ACTIVATION_DTYPE frag_k[hold_size];
        // Declare Shared Memory for all aligned to 16 bytes.
        __shared__ __align__(16) unsigned char raw_smem_Q[Q_TILE_ROWS * DIM_HEAD * sizeof(ACTIVATION_DTYPE)];
        __shared__ __align__(16) unsigned char raw_smem_K[2 * (KV_TILE_ROWS * DIM_HEAD * sizeof(ACTIVATION_DTYPE))];
        __shared__ __align__(16) unsigned char raw_smem_V[2 * (KV_TILE_ROWS * DIM_HEAD * sizeof(ACTIVATION_DTYPE))];
        __shared__ __align__(16) unsigned char raw_smem_O[2 * (Q_TILE_ROWS * DIM_HEAD * sizeof(ACTIVATION_DTYPE))];
        __shared__ __align__(16) float smem_L[Q_TILE_ROWS];
        __shared__ __align__(16) float smem_M[Q_TILE_ROWS];
        ACTIVATION_DTYPE* smem_Q = reinterpret_cast<ACTIVATION_DTYPE*>(raw_smem_Q);
        ACTIVATION_DTYPE* smem_K = reinterpret_cast<ACTIVATION_DTYPE*>(raw_smem_K);
        ACTIVATION_DTYPE* smem_V = reinterpret_cast<ACTIVATION_DTYPE*>(raw_smem_V);
        ACTIVATION_DTYPE* smem_O = reinterpret_cast<ACTIVATION_DTYPE*>(raw_smem_O);

        auto block = cooperative_groups::this_thread_block();
        __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> shared_pipe_state;
        auto pipe = cuda::make_pipeline(block, &shared_pipe_state);
        int buffer_iter{0};

        // Update pointers to start outside of loop.
        // Update Q once (since one Q tile per worker) across batch items, heads, tiles to start of tile.
        // Memcpy one contiguous row of Q tile per thread as applicable.
        Q += blockIdx.x * stride_batch_q + blockIdx.y * stride_head_dim_q + blockIdx.z * Q_TILE_ROWS * DIM_HEAD;
        pipe.producer_acquire();
        if (block.thread_rank() < Q_TILE_ROWS) {
            cuda::memcpy_async(smem_Q + block.thread_rank() * DIM_HEAD, Q + block.thread_rank() * blockDim.y * DIM_HEAD, DIM_HEAD * sizeof(ACTIVATION_DTYPE), pipe);
        }
        pipe.producer_commit();
        
        // Start within first KV subtile since iterate through all of them.
        int kv_tile_offset = blockIdx.x * stride_batch_kv + blockIdx.y * stride_head_dim_kv;
        K += kv_tile_offset;
        V += kv_tile_offset;
        
        // Launch pre-emptive load of first KV immediately. Same as Q.
        pipe.producer_acquire();
        if (block.thread_rank() < KV_TILE_ROWS) {
            cuda::memcpy_async(smem_K + block.thread_rank() * DIM_HEAD, K + block.thread_rank() * blockDim.y * DIM_HEAD, DIM_HEAD * sizeof(ACTIVATION_DTYPE), pipe);
            cuda::memcpy_async(smem_V + block.thread_rank() * DIM_HEAD, V + block.thread_rank() * blockDim.y * DIM_HEAD, DIM_HEAD * sizeof(ACTIVATION_DTYPE), pipe);
        }
        pipe.producer_commit();

        // Wait on Q to be loaded to Smem.
        pipe.consumer_wait();
        block.sync();
        pipe.consumer_release();

        for (int kv_tile_row{0}; kv_tile_row < seq_len; kv_tile_row += KV_TILE_ROWS) {
            // If last q tile row NOT less the first KV tile row: Do matmul
            if ((blockIdx.z + 1) * Q_TILE_ROWS - 1 >= kv_tile) {
                K += kv_tile_row * blockDim.y * DIM_HEAD;
                V += kv_tile_row * blockDim.y * DIM_HEAD;
                // Init L and M.
                memset(smem_L, 0, Q_TILE_ROWS * sizeof(ACTIVATION_DTYPE));
                memset(smem_M, -INFINITY, Q_TILE_ROWS * sizeof(ACTIVATION_DTYPE));

                // Launch async loads of next KV tiles.
                pipe.producer_acquire();
                if (block.thread_rank() < KV_TILE_ROWS) {
                    cuda::memcpy_async(smem_K + (buffer_iter^=1) * block.thread_rank() * DIM_HEAD, K + (block.thread_rank() + kv_tile_row) * blockDim.y * DIM_HEAD, DIM_HEAD * sizeof(ACTIVATION_DTYPE), pipe);
                    cuda::memcpy_async(smem_V + (buffer_iter^=1) * block.thread_rank() * DIM_HEAD, V + (block.thread_rank() + kv_tile_row) * blockDim.y * DIM_HEAD, DIM_HEAD * sizeof(ACTIVATION_DTYPE), pipe);
                }
                pipe.producer_commit();

                // Await currently needed KV Tile
                pipe.consumer_wait();
                pipe.consumer_release();
                
                // Wraps work left to right horizontally across Q and vertically across K and workers stacked vertically along Q. warp_id * 16 rows per subtile * DIM_HEAD per row.
                // Each subtile output will be the small scores output.
                int row_offset = threadIdx.x * 16 * DIM_HEAD;
                for (int horizontal_q_subtile_col{0}; horizontal_q_subtile_col < DIM_HEAD; horizontal_q_subtile_col += 16) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, ACTIVATION_DTYPE, wmma::row_major> a_frag;
                    wmma::fragment<wmma::accumulator, 16, 16, 16, ACTIVATION_DTYPE> accum_frag;
                    // Grab from correct buffer, correct top left corner's row, correct top left corner's col
                    wmma::load_matrix_sync(a_frag, smem_Q + buffer_iter * Q_TILE_ROWS * DIM_HEAD + threadIdx.x * 16 * DIM_HEAD + horizontal_q_subtile_col, DIM_HEAD);
                    for (int vertical_subtile_kv_row{0}; subtile < KV_TILE_ROWS; subtile += 16) {
                        wmma::fill_fragment(accum_frag, ACTIVATION_DTYPE(0.0f));
                        wmma::fragment<wmma::matrix_b, 16, 16, 16, ACTIVATION_DTYPE, wmma::col_major> b_frag;
                        // Correct buffer, correct row, correct col
                        wmma::load_matrix_sync(b_frag, smem_K + buffer_iter * KV_TILE_ROWS * DIM_HEAD + vertical_subtile_kv_row + threadIdx.x * 16, DIM_HEAD);
                        wmma::mma_sync(accum_frag, a_frag, b_frag, accum_frag);
                        ACTIVATION_DTYPE thread_scores[8];
                        for (int i = 0; i < 8; ++i) {
                            thread_scores[i] = accum_frag.x[threadIdx.y * 16 + threadIdx.z * 8 + i];
                        }
                    }
                }
            }
        }
    }
}

}