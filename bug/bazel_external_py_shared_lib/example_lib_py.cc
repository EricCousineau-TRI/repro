#include <pybind11/pybind11.h>

#include "example_shared.h"

PYBIND11_MODULE(example_lib_py, m) {
  m.def("func", &func);
}
