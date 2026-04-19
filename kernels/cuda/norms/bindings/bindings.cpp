#include <torch/extension.h>
#include <pybind11/operators.h>
#include "launchers.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    using namespace pybind11::literals;
}
