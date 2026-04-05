#include "../norms/rmsnorm.cuh"
#include <torch/extension.h>
#include <pybind11/operators.h>

// -------------------------------------
// Shared dim switch cases
// -------------------------------------

#define RMSNORM_DIM_CASES(ACT, WGT, VEC, CALC)                                              \
    case 128:   return norms::launchers::offline::cuda_rmsnorm_divisble_forward_launcher_offline<ACT, WGT, VEC, CALC, 128>(x, gamma, epsilon, num_threads);   \
    /*case 256:   return norms::launchers::offline::cuda_rmsnorm_divisble_forward_launcher_offline<ACT, WGT, VEC, CALC, 256>(x, gamma, epsilon, num_threads);   \
    case 384:   return norms::launchers::offline::cuda_rmsnorm_divisble_forward_launcher_offline<ACT, WGT, VEC, CALC, 384>(x, gamma, epsilon, num_threads);   \
    case 512:   return norms::launchers::offline::cuda_rmsnorm_divisble_forward_launcher_offline<ACT, WGT, VEC, CALC, 512>(x, gamma, epsilon, num_threads);   \
    case 768:   return norms::launchers::offline::cuda_rmsnorm_divisble_forward_launcher_offline<ACT, WGT, VEC, CALC, 768>(x, gamma, epsilon, num_threads);   \
    case 1024:  return norms::launchers::offline::cuda_rmsnorm_divisble_forward_launcher_offline<ACT, WGT, VEC, CALC, 1024>(x, gamma, epsilon, num_threads);  \
    case 1536:  return norms::launchers::offline::cuda_rmsnorm_divisble_forward_launcher_offline<ACT, WGT, VEC, CALC, 1536>(x, gamma, epsilon, num_threads);  \
    case 2048:  return norms::launchers::offline::cuda_rmsnorm_divisble_forward_launcher_offline<ACT, WGT, VEC, CALC, 2048>(x, gamma, epsilon, num_threads);  \
    case 5120:  return norms::launchers::offline::cuda_rmsnorm_divisble_forward_launcher_offline<ACT, WGT, VEC, CALC, 5120>(x, gamma, epsilon, num_threads);  \
    case 10240: return norms::launchers::offline::cuda_rmsnorm_divisble_forward_launcher_offline<ACT, WGT, VEC, CALC, 10240>(x, gamma, epsilon, num_threads) */

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
// -------------------------------------

DEFINE_RMSNORM_ENTRY(rmsnorm_fp32_fp32_f4,   float,           float,           float4, float)
DEFINE_RMSNORM_ENTRY(rmsnorm_fp32_fp32_f2,   float,           float,           float2, float)
DEFINE_RMSNORM_ENTRY(rmsnorm_fp32_fp32_f1,   float,           float,           float,  float)

DEFINE_RMSNORM_ENTRY(rmsnorm_fp16_fp16_f4,   at::Half,        at::Half,        float4, float)
DEFINE_RMSNORM_ENTRY(rmsnorm_fp16_fp16_f2,   at::Half,        at::Half,        float2, float)
DEFINE_RMSNORM_ENTRY(rmsnorm_fp16_fp16_f1,   at::Half,        at::Half,        float,  float)

DEFINE_RMSNORM_ENTRY(rmsnorm_fp16_fp32_f4,   at::Half,        float,           float4, float)
DEFINE_RMSNORM_ENTRY(rmsnorm_fp16_fp32_f2,   at::Half,        float,           float2, float)
DEFINE_RMSNORM_ENTRY(rmsnorm_fp16_fp32_f1,   at::Half,        float,           float,  float)

DEFINE_RMSNORM_ENTRY(rmsnorm_bf16_bf16_f4,   at::BFloat16,    at::BFloat16,    float4, float)
DEFINE_RMSNORM_ENTRY(rmsnorm_bf16_bf16_f2,   at::BFloat16,    at::BFloat16,    float2, float)
DEFINE_RMSNORM_ENTRY(rmsnorm_bf16_bf16_f1,   at::BFloat16,    at::BFloat16,    float,  float)

DEFINE_RMSNORM_ENTRY(rmsnorm_bf16_fp32_f4,   at::BFloat16,    float,           float4, float)
DEFINE_RMSNORM_ENTRY(rmsnorm_bf16_fp32_f2,   at::BFloat16,    float,           float2, float)
DEFINE_RMSNORM_ENTRY(rmsnorm_bf16_fp32_f1,   at::BFloat16,    float,           float,  float)

// -------------------------------------
// Pybind
// Direct exported entries, no type dispatch
// -------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    using namespace pybind11::literals;
    m.def("rmsnorm_fp32_fp32_f4", &rmsnorm_fp32_fp32_f4, "x"_a, "gamma"_a, "epsilon"_a=1e-5f, "num_threads"_a=256);
    m.def("rmsnorm_fp32_fp32_f2", &rmsnorm_fp32_fp32_f2, "x"_a, "gamma"_a, "epsilon"_a=1e-5f, "num_threads"_a=256);
    m.def("rmsnorm_fp32_fp32_f1", &rmsnorm_fp32_fp32_f1, "x"_a, "gamma"_a, "epsilon"_a=1e-5f, "num_threads"_a=256);

    m.def("rmsnorm_fp16_fp16_f4", &rmsnorm_fp16_fp16_f4, "x"_a, "gamma"_a, "epsilon"_a=1e-5f, "num_threads"_a=256);
    m.def("rmsnorm_fp16_fp16_f2", &rmsnorm_fp16_fp16_f2, "x"_a, "gamma"_a, "epsilon"_a=1e-5f, "num_threads"_a=256);
    m.def("rmsnorm_fp16_fp16_f1", &rmsnorm_fp16_fp16_f1, "x"_a, "gamma"_a, "epsilon"_a=1e-5f, "num_threads"_a=256);

    m.def("rmsnorm_fp16_fp32_f4", &rmsnorm_fp16_fp32_f4, "x"_a, "gamma"_a, "epsilon"_a=1e-5f, "num_threads"_a=256);
    m.def("rmsnorm_fp16_fp32_f2", &rmsnorm_fp16_fp32_f2, "x"_a, "gamma"_a, "epsilon"_a=1e-5f, "num_threads"_a=256);
    m.def("rmsnorm_fp16_fp32_f1", &rmsnorm_fp16_fp32_f1, "x"_a, "gamma"_a, "epsilon"_a=1e-5f, "num_threads"_a=256);

    m.def("rmsnorm_bf16_bf16_f4", &rmsnorm_bf16_bf16_f4, "x"_a, "gamma"_a, "epsilon"_a=1e-5f, "num_threads"_a=256);
    m.def("rmsnorm_bf16_bf16_f2", &rmsnorm_bf16_bf16_f2, "x"_a, "gamma"_a, "epsilon"_a=1e-5f, "num_threads"_a=256);
    m.def("rmsnorm_bf16_bf16_f1", &rmsnorm_bf16_bf16_f1, "x"_a, "gamma"_a, "epsilon"_a=1e-5f, "num_threads"_a=256);

    m.def("rmsnorm_bf16_fp32_f4", &rmsnorm_bf16_fp32_f4, "x"_a, "gamma"_a, "epsilon"_a=1e-5f, "num_threads"_a=256);
    m.def("rmsnorm_bf16_fp32_f2", &rmsnorm_bf16_fp32_f2, "x"_a, "gamma"_a, "epsilon"_a=1e-5f, "num_threads"_a=256);
    m.def("rmsnorm_bf16_fp32_f1", &rmsnorm_bf16_fp32_f1, "x"_a, "gamma"_a, "epsilon"_a=1e-5f, "num_threads"_a=256);
}

// -------------------------------------
// Cleanup
// -------------------------------------

#undef DEFINE_RMSNORM_ENTRY
#undef RMSNORM_DIM_CASES