# Shared Memory

This is an on-chip memory cache that takes less cycles to access versus HBM (global memory). It is something like 5x faster to access and bandwidth. Memory-bound operations (like Attention) benefit massively from correctly implementing memory caching.

The typicall max SMEM per thread block is 100Kb, 160Kb, and 227 Kb for the RTX 4090, A100, and H100 respectively.

## Static Shared Memory

The amount of smem (size) needed is known at compile time. This is declared inside the kernel with a fixed size (__shared T smem[256];). When this this kernel is compiled, the compiler allocated that smem and is fixed for every block. This will appear in the compiled metadata, so when you use --ptxas-options=-v, you will see that it requires N bytes of smem.

## Dynamic Shared Memory

The amount of smem needed is NOT known at compile time and is instead decided at kernel launch. This is declared using extern and the suze is determined when passing launch parameters. The third kernel launch parameter becomes how much smem to allocate kernel<<<grid, block, smem_size>>>(input);.

Many high performance kernels typically use dynamic SMEM because the tile size is dependent on the GPU architecture, rgister pressure, and tile size.

Also an incredibly important performance note: The only thing that changes with dynamic memory is the address. The perf is almost exactly identical to static. That's why dynamic is used almost exclusively in CUTLASS, Triton, FlashAttention, and DeepSpeed.

Dynamic also has another advantage: it can exceed the default per-block SMEM (typically 48KB) and go to almost the max per SM (about 2x larger if not more than default). You cannot, however, exceed the max per SM limit for obvious reasons.

## Miscellaneous Info

When writing a kernel that uses type T and you want to use SMEM, that means that smem needs to be of type T (extern __shared__ T smem_data[];), but this is not allowed.
One solution is to just declare extern __shared__ char smem[]; T* smem_data = reinterpret_cast<T*>(smem);
