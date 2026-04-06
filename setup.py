from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="ModelComponentsAndKernels", # Project name (doesn't really do anything lol, just the pip install)
    ext_package="ComponentsAndKernels",
    ext_modules=[
        CUDAExtension(
            name="Kernels.fused_norms", # Compiled extension module (import path)
            sources=[
                "kernels/cuda/norms/bindings/bindings.cpp",
                "kernels/cuda/norms/src/rmsnorm.cu"
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-lineinfo", "--use_fast_math"], # --use_fast_math, -Xptas=-v, --expt-relaxed-constexpr
            }
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)