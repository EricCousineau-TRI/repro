#include "pybind11/pybind11.h"

#include "example_lib.h"

PYBIND11_MODULE(example_lib_py, m) {
  m.def("func", &func);
}
