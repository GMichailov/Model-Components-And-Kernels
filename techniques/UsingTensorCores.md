# Tensor Cores

Tensor cores are dedicated acclerators for GEMM operations. These are built specifically to accelerate
matrix multiply-accumulate (MMA) operations (D = AB + C). These can perform one MMA per cycle rather than 
needing to do thousands of scalar ops like CUDA cores. A warp cooperatively executed a matmul instruction 
called mma.sync where each thread holds fragments in registers. These cores contain massive grids for this.
These tensor cores also use reduced precision types. The throughput can be something like 10-20x higher than
normal CUDA cores for matrix math.

## Column Major Example

Variables:
- T1\

```C++
template <typename T1, typename T2, int M, int N, int K, typename WMMA_FRAG_LAYOUT_A, typename WMMA_FRAG_LAYOUT_B>
__global__ void wmma_gemm_a_col_major_b_col_major(
  T1 const* A, T1 const* B, T1 const* C, T2* D, uint32_t m, uint32_t n, uint32_t k,
  uint32_t lda, uint32_t ldb, uint32_t ldc, bool is_A_tranpose, bool is_B_tranpose, float alpha, float beta
) {
  // MMA instructions are issued per warp. This is so that each thread knows which warp it exists in in the X and Y directions.
  uint32_t const warpM{blockIdx.x * blockDim.x + threadIdx.x) / 32; // 32 being number of threads in a warp.
  uint32_t const warpN{blockIdx.y * blockDim.y + threadIdx.y);

  // Declare fragments.
  // Fragments are a small tile of the global matrix stored in registers.
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M, N, K, T1, WMMA_FRAG_LAYOUT_A> frag_a{};
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M, N, K, T1, WMMA_FRAG_LAYOUT_B> frag_b{};
  // C
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M, N, K, T1, WMMA_FRAG_LAYOUT_A> frag_a{};
}
