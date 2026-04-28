#include <torch/extension.h>
#include <pybind11/operators.h>
#include "launchers.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    using namespace pybind11::literals;

    m.def("rmsnorm_online_bf16_bf16_f32_t32", &rmsnorm_online_bf16_bf16_f32_t32, "x"_a, "gamma"_a, "epsilon"_a, "dim_x"_a);
}
