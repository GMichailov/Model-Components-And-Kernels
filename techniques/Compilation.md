# Compiling for Different Architectures

Host code is compiled by the C++ compiler, however, device code is compiled in a specific way for different architectures.
CUDA uses arch macros so code came behave differently for each GPU type to prevent subtle bugs.

## Compute Capability Differences

Different GPUs have support for different features such as Tensor Core versions, async copy instructions, SMEM size, warp-level primitives, FP8 support, sparsity acceleration.
Code often needs 
#if __CUDA_ARCH__ >= 900
  // Hopper Implementation
#elif __CUDA_ARCH__ >= 860
  // RTX
#elif __CUDA_ARCH__ >= 800
  // A100
#endif

## PTX vs SASS Compatibility

SASS is native GPU machine code which is specific to the GPU architecture and PTX is a lower-level virtual GPU assembly that can JIT-compile.
The very, very important caveat is that while PTX is specifically made to be able to compile for more than one arch, it is typically meant for FORWARD (latter GPU archs) not for previous.
So for example, a CUDA_ARCH = 800 (A100) PTX kernel compilation will be usable for the H100, but significantly less effective because it can't leverage the benefits offered by next gen GPUs.

PyTorch ships multiple cubins and the driver auto picks the best matching image. If there is an exact SASS match, use that, otherwise, use the best PTX JIT if possible.

## Driver/Toolkit compatibility

Binaries built with CUDA 12.x may not run on systems with older drivers or may lack support for newer instructions which is painfully common.
