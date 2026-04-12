# Tensor Cores

Tensor cores are dedicated acclerators for GEMM operations. These are built specifically to accelerate
matrix multiply-accumulate (MMA) operations (D = AB + C). These can perform one MMA per cycle rather than 
needing to do thousands of scalar ops like CUDA cores. A warp cooperatively executed a matmul instruction 
called mma.sync where each thread holds fragments in registers. These cores contain massive grids for this.
These tensor cores also use reduced precision types. The throughput can be something like 10-20x higher than
normal CUDA cores for matrix math.

## Column Major Example

This example uses D = alpha * A * B + beta * C

Variables:
- T1: The data type of matrices A, B, and C
- T2: The data type of matrices D
- M, N, K are the WMMA tile sizes (the sizes inside memory). These are typically 16.
- WMMA_FRAG_LAYOUT_A/B: These are either nvcuda::wmma::(row_major/col_major). It tells WMMA how to interpret the memory layout.
- A, B, C, D are pointers to the matrices.
- m, n, k are the dimensions of the global matrices.
- lda, ldb, ldc, ldd: This means the distance between column starts in memory.
- Transpose flags tell if that respective matrix should be treated as transposed.

```C++
template <typename T1, typename T2, int M, int N, int K, typename WMMA_FRAG_LAYOUT_A, typename WMMA_FRAG_LAYOUT_B, typename WMMA_FRAG_LAYOUT_C>
__global__ void wmma_gemm_a_col_major_b_col_major(
  T1 const* A, T1 const* B, T1 const* C, T2* D, uint32_t m, uint32_t n, uint32_t k,
  uint32_t lda, uint32_t ldb, uint32_t ldc, bool is_A_tranpose, bool is_B_tranpose, float alpha, float beta
) {
  // MMA instructions are issued per warp. This is so that each thread knows which warp it exists in in the X and Y directions.
  uint32_t const warpM{blockIdx.x * blockDim.x + threadIdx.x) / 32; // 32 being number of threads in a warp.
  uint32_t const warpN{blockIdx.y * blockDim.y + threadIdx.y);

  // Declare fragments.
  // Fragments are a small tile of the global matrix stored in registers.
  // The frag_accum fragment is the running sum. (Since this will compute several tile GEMMs one after each other).
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M, N, K, T1, WMMA_FRAG_LAYOUT_A> frag_a{};
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M, N, K, T1, WMMA_FRAG_LAYOUT_B> frag_b{};
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_c, M, N, K, T1, WMMA_FRAG_LAYOUT_C> frag_c{};
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, T2> frag_accum{};
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, T2> frag_d{};
  // Make accumulator start from 0.
  nvcuda::wmma::fill_fragment(frag_accum, static_cast<T2>(0));
  // Loop over K
  for (uint32_t ki{0}; ki < k; ki += K) {
    uint32_t const matrix_mma_a_row_idx{is_A_transpose ? ki : warpM * M};
    uint32_t const matrix_mma_a_col_idx{is_A_transpose ? warpM * M : ki};
    uint32_t const matrix_mma_b_row_idx{is_B_transpose ? ki : warpN * N};
    uint32_t const matrix_mma_b_col_idx{is_B_transpose ? warpN * N : ki};

    // Check bounds
    if(matrix_mma_a_row_idx < (is_A_tranpose ? k : m) &&
       matrix_mma_a_col_idx < (is_A_tranpose ? m : k) &&
       matrix_mma_b_row_idx < (is_B_tranpose ? n : k) &&
       matrix_mma_b_col_idx < (is_B_tranpose ? k : n)) {
      T1 const* matrix_mma_a_mptr{A + matrix_mma_a_row_idx + matrix_mma_a_col_idx * lda};
      T1 const* matrix_mma_b_mptr{B + matrix_mma_b_row_idx + matrix_mma_b_col_idx * ldb};
      nvcuda::wmma::load_matrix_sync(frag_a, matrix_mma_a_mptr, lda);
      nvcuda::wmma::load_matrix_sync(frag_b, matrix_mma_b_mptr, ldb);
    }
  }
}
