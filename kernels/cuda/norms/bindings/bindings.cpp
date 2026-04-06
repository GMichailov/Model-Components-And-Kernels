#include <torch/extension.h>
#include <pybind11/operators.h>
#include "launchers.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    using namespace pybind11::literals;

    m.def("rmsnorm_forward_fp32_fp32_fp32_v1", &rmsnorm_forward_fp32_fp32_fp32_v1, "x"_a, "gamma"_a, "epsilon"_a, "num_threads"_a);
    m.def("rmsnorm_forward_fp32_fp32_fp32_v2", &rmsnorm_forward_fp32_fp32_fp32_v2, "x"_a, "gamma"_a, "epsilon"_a, "num_threads"_a);
    m.def("rmsnorm_forward_fp32_fp32_fp32_v4", &rmsnorm_forward_fp32_fp32_fp32_v4, "x"_a, "gamma"_a, "epsilon"_a, "num_threads"_a);
    m.def("rmsnorm_forward_fp16_fp16_fp32_v1", &rmsnorm_forward_fp16_fp16_fp32_v1, "x"_a, "gamma"_a, "epsilon"_a, "num_threads"_a);
    m.def("rmsnorm_forward_fp16_fp16_fp32_v2", &rmsnorm_forward_fp16_fp16_fp32_v2, "x"_a, "gamma"_a, "epsilon"_a, "num_threads"_a);
    m.def("rmsnorm_forward_fp16_fp16_fp32_v4", &rmsnorm_forward_fp16_fp16_fp32_v4, "x"_a, "gamma"_a, "epsilon"_a, "num_threads"_a);
    m.def("rmsnorm_forward_fp16_fp16_fp32_v8", &rmsnorm_forward_fp16_fp16_fp32_v8, "x"_a, "gamma"_a, "epsilon"_a, "num_threads"_a);
    m.def("rmsnorm_forward_fp16_fp32_fp32_v1", &rmsnorm_forward_fp16_fp32_fp32_v1, "x"_a, "gamma"_a, "epsilon"_a, "num_threads"_a);
    m.def("rmsnorm_forward_fp16_fp32_fp32_v2", &rmsnorm_forward_fp16_fp32_fp32_v2, "x"_a, "gamma"_a, "epsilon"_a, "num_threads"_a);
    m.def("rmsnorm_forward_fp16_fp32_fp32_v4", &rmsnorm_forward_fp16_fp32_fp32_v4, "x"_a, "gamma"_a, "epsilon"_a, "num_threads"_a);
    m.def("rmsnorm_forward_fp16_fp32_fp32_v8", &rmsnorm_forward_fp16_fp32_fp32_v8, "x"_a, "gamma"_a, "epsilon"_a, "num_threads"_a);
    m.def("rmsnorm_forward_bf16_bf16_fp32_v1", &rmsnorm_forward_bf16_bf16_fp32_v1, "x"_a, "gamma"_a, "epsilon"_a, "num_threads"_a);
    m.def("rmsnorm_forward_bf16_bf16_fp32_v2", &rmsnorm_forward_bf16_bf16_fp32_v2, "x"_a, "gamma"_a, "epsilon"_a, "num_threads"_a);
    m.def("rmsnorm_forward_bf16_bf16_fp32_v4", &rmsnorm_forward_bf16_bf16_fp32_v4, "x"_a, "gamma"_a, "epsilon"_a, "num_threads"_a);
    m.def("rmsnorm_forward_bf16_bf16_fp32_v8", &rmsnorm_forward_bf16_bf16_fp32_v8, "x"_a, "gamma"_a, "epsilon"_a, "num_threads"_a);
    m.def("rmsnorm_forward_bf16_fp32_fp32_v1", &rmsnorm_forward_bf16_fp32_fp32_v1, "x"_a, "gamma"_a, "epsilon"_a, "num_threads"_a);
    m.def("rmsnorm_forward_bf16_fp32_fp32_v2", &rmsnorm_forward_bf16_fp32_fp32_v2, "x"_a, "gamma"_a, "epsilon"_a, "num_threads"_a);
    m.def("rmsnorm_forward_bf16_fp32_fp32_v4", &rmsnorm_forward_bf16_fp32_fp32_v4, "x"_a, "gamma"_a, "epsilon"_a, "num_threads"_a);
    m.def("rmsnorm_forward_bf16_fp32_fp32_v8", &rmsnorm_forward_bf16_fp32_fp32_v8, "x"_a, "gamma"_a, "epsilon"_a, "num_threads"_a);
}
