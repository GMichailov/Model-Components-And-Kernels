from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="Kernels",
    ext_modules=[
        CUDAExtension(
            name="fused_norms",
            sources=[
                "kernels/cuda/bindings/bindings.cpp",
                "kernels/cuda/norms/rmsnorm.cu"
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-lineinfo", "--use_fast_math"], # --use_fast_math, -Xptas=-v, --expt-relaxed-constexpr
            }
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)