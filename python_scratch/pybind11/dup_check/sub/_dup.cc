#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using std::string;

string get_name() {
    return "sub level";
}

PYBIND11_PLUGIN(_dup) {
  py::module m("_dup",
               "Dup at sub level.");

  m.def("get_name", &get_name);

  return m.ptr();
}
