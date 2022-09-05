#include <string>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "common/a.h"
#include "common/b.h"

namespace py = pybind11;

namespace example {

PYBIND11_MODULE(cc, m) {
  m.doc() = "Bindings for //common";

  // For logging redirects.
  py::module::import("pydrake.common");

  m.def("FuncA", &FuncA);
  m.def("FuncB", &FuncB);
}

}  // namespace example
