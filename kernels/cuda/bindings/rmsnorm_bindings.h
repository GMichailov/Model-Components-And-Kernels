#include "../norms/rmsnorm.cuh"
#include <torch/extension.h>
#include <pybind11/operators.h>

// -------------------------------------
// Shared dim switch cases
// -------------------------------------

#define RMSNORM_DIM_CASES(ACT, WGT, VEC, CALC)                                              \
    case 128:   return norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<ACT, WGT, CALC, VEC, 128>(x, gamma, epsilon, num_threads);   \
    /*case 256:   return norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<ACT, WGT, CALC, VEC, 256>(x, gamma, epsilon, num_threads);   \
    case 384:   return norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<ACT, WGT, CALC, VEC, 384>(x, gamma, epsilon, num_threads);   \
    case 512:   return norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<ACT, WGT, CALC, VEC, 512>(x, gamma, epsilon, num_threads);   \
    case 768:   return norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<ACT, WGT, CALC, VEC, 768>(x, gamma, epsilon, num_threads);   \
    case 1024:  return norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<ACT, WGT, CALC, VEC, 1024>(x, gamma, epsilon, num_threads);  \
    case 1536:  return norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<ACT, WGT, CALC, VEC, 1536>(x, gamma, epsilon, num_threads);  \
    case 2048:  return norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<ACT, WGT, CALC, VEC, 2048>(x, gamma, epsilon, num_threads);  \
    case 5120:  return norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<ACT, WGT, CALC, VEC, 5120>(x, gamma, epsilon, num_threads);  \
    case 10240: return norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline<ACT, WGT, CALC, VEC, 10240>(x, gamma, epsilon, num_threads) */

// -------------------------------------
// Entry generator
// One entrypoint per concrete type config
// Only switches on dim
// -------------------------------------

#define DEFINE_RMSNORM_ENTRY(FN_NAME, ACT, WGT, VEC, CALC)                  \
at::Tensor FN_NAME(                                                      \
    const at::Tensor& x,                                                 \
    const at::Tensor& gamma,                                             \
    float epsilon,                                                          \
    int num_threads                                                         \
) {                                                                         \
    const int dim = static_cast<int>(x.size(-1));                           \
    switch (dim) {                                                          \
        RMSNORM_DIM_CASES(ACT, WGT, VEC, CALC);                             \
        default:                                                            \
            throw std::runtime_error("Unsupported dim for " #FN_NAME);      \
    }                                                                       \
}

// -------------------------------------
// Concrete direct entries
// VEC is the vectorized load COUNT (1, 2, 4, 8 elements)
// -------------------------------------

// fp32, fp32 - vec counts 1, 2, 4
DEFINE_RMSNORM_ENTRY(rmsnorm_fp32_fp32_v1, float, float, 1, float)
DEFINE_RMSNORM_ENTRY(rmsnorm_fp32_fp32_v2, float, float, 2, float)
DEFINE_RMSNORM_ENTRY(rmsnorm_fp32_fp32_v4, float, float, 4, float)

// fp16, fp16 - vec counts 1, 2, 4, 8
DEFINE_RMSNORM_ENTRY(rmsnorm_fp16_fp16_v1, at::Half, at::Half, 1, float)
DEFINE_RMSNORM_ENTRY(rmsnorm_fp16_fp16_v2, at::Half, at::Half, 2, float)
DEFINE_RMSNORM_ENTRY(rmsnorm_fp16_fp16_v4, at::Half, at::Half, 4, float)
DEFINE_RMSNORM_ENTRY(rmsnorm_fp16_fp16_v8, at::Half, at::Half, 8, float)

// fp16, fp32 - vec counts 1, 2, 4, 8
DEFINE_RMSNORM_ENTRY(rmsnorm_fp16_fp32_v1, at::Half, float, 1, float)
DEFINE_RMSNORM_ENTRY(rmsnorm_fp16_fp32_v2, at::Half, float, 2, float)
DEFINE_RMSNORM_ENTRY(rmsnorm_fp16_fp32_v4, at::Half, float, 4, float)
DEFINE_RMSNORM_ENTRY(rmsnorm_fp16_fp32_v8, at::Half, float, 8, float)

// bf16, bf16 - vec counts 1, 2, 4, 8
DEFINE_RMSNORM_ENTRY(rmsnorm_bf16_bf16_v1, at::BFloat16, at::BFloat16, 1, float)
DEFINE_RMSNORM_ENTRY(rmsnorm_bf16_bf16_v2, at::BFloat16, at::BFloat16, 2, float)
DEFINE_RMSNORM_ENTRY(rmsnorm_bf16_bf16_v4, at::BFloat16, at::BFloat16, 4, float)
DEFINE_RMSNORM_ENTRY(rmsnorm_bf16_bf16_v8, at::BFloat16, at::BFloat16, 8, float)

// bf16, fp32 - vec counts 1, 2, 4, 8
DEFINE_RMSNORM_ENTRY(rmsnorm_bf16_fp32_v1, at::BFloat16, float, 1, float)
DEFINE_RMSNORM_ENTRY(rmsnorm_bf16_fp32_v2, at::BFloat16, float, 2, float)
DEFINE_RMSNORM_ENTRY(rmsnorm_bf16_fp32_v4, at::BFloat16, float, 4, float)
DEFINE_RMSNORM_ENTRY(rmsnorm_bf16_fp32_v8, at::BFloat16, float, 8, float)

// -------------------------------------
// Pybind
// Direct exported entries
// -------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    using namespace pybind11::literals;

    // fp32, fp32
    m.def("rmsnorm_fp32_fp32_v1", &rmsnorm_fp32_fp32_v1, "x"_a, "gamma"_a, "epsilon"_a=1e-5f, "num_threads"_a=256);
    m.def("rmsnorm_fp32_fp32_v2", &rmsnorm_fp32_fp32_v2, "x"_a, "gamma"_a, "epsilon"_a=1e-5f, "num_threads"_a=256);
    m.def("rmsnorm_fp32_fp32_v4", &rmsnorm_fp32_fp32_v4, "x"_a, "gamma"_a, "epsilon"_a=1e-5f, "num_threads"_a=256);

    // fp16, fp16
    m.def("rmsnorm_fp16_fp16_v1", &rmsnorm_fp16_fp16_v1, "x"_a, "gamma"_a, "epsilon"_a=1e-5f, "num_threads"_a=256);
    m.def("rmsnorm_fp16_fp16_v2", &rmsnorm_fp16_fp16_v2, "x"_a, "gamma"_a, "epsilon"_a=1e-5f, "num_threads"_a=256);
    m.def("rmsnorm_fp16_fp16_v4", &rmsnorm_fp16_fp16_v4, "x"_a, "gamma"_a, "epsilon"_a=1e-5f, "num_threads"_a=256);
    m.def("rmsnorm_fp16_fp16_v8", &rmsnorm_fp16_fp16_v8, "x"_a, "gamma"_a, "epsilon"_a=1e-5f, "num_threads"_a=256);

    // fp16, fp32
    m.def("rmsnorm_fp16_fp32_v1", &rmsnorm_fp16_fp32_v1, "x"_a, "gamma"_a, "epsilon"_a=1e-5f, "num_threads"_a=256);
    m.def("rmsnorm_fp16_fp32_v2", &rmsnorm_fp16_fp32_v2, "x"_a, "gamma"_a, "epsilon"_a=1e-5f, "num_threads"_a=256);
    m.def("rmsnorm_fp16_fp32_v4", &rmsnorm_fp16_fp32_v4, "x"_a, "gamma"_a, "epsilon"_a=1e-5f, "num_threads"_a=256);
    m.def("rmsnorm_fp16_fp32_v8", &rmsnorm_fp16_fp32_v8, "x"_a, "gamma"_a, "epsilon"_a=1e-5f, "num_threads"_a=256);

    // bf16, bf16
    m.def("rmsnorm_bf16_bf16_v1", &rmsnorm_bf16_bf16_v1, "x"_a, "gamma"_a, "epsilon"_a=1e-5f, "num_threads"_a=256);
    m.def("rmsnorm_bf16_bf16_v2", &rmsnorm_bf16_bf16_v2, "x"_a, "gamma"_a, "epsilon"_a=1e-5f, "num_threads"_a=256);
    m.def("rmsnorm_bf16_bf16_v4", &rmsnorm_bf16_bf16_v4, "x"_a, "gamma"_a, "epsilon"_a=1e-5f, "num_threads"_a=256);
    m.def("rmsnorm_bf16_bf16_v8", &rmsnorm_bf16_bf16_v8, "x"_a, "gamma"_a, "epsilon"_a=1e-5f, "num_threads"_a=256);

    // bf16, fp32
    m.def("rmsnorm_bf16_fp32_v1", &rmsnorm_bf16_fp32_v1, "x"_a, "gamma"_a, "epsilon"_a=1e-5f, "num_threads"_a=256);
    m.def("rmsnorm_bf16_fp32_v2", &rmsnorm_bf16_fp32_v2, "x"_a, "gamma"_a, "epsilon"_a=1e-5f, "num_threads"_a=256);
    m.def("rmsnorm_bf16_fp32_v4", &rmsnorm_bf16_fp32_v4, "x"_a, "gamma"_a, "epsilon"_a=1e-5f, "num_threads"_a=256);
    m.def("rmsnorm_bf16_fp32_v8", &rmsnorm_bf16_fp32_v8, "x"_a, "gamma"_a, "epsilon"_a=1e-5f, "num_threads"_a=256);
}

// -------------------------------------
// Cleanup
// -------------------------------------

#undef DEFINE_RMSNORM_ENTRY
#undef RMSNORM_DIM_CASES
