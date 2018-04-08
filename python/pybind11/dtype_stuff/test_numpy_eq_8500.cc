#include <iostream>

#include <pybind11/embed.h>

namespace py = pybind11;

using std::cout;
using std::endl;

int main() {
  py::scoped_interpreter guard;

  py::module m("__main__");
  m.def("trigger", []() {
    py::print("Triggered");
  });

  py::str file = "python/pybind11/dtype_stuff/test_numpy_eq_8500.py";
  m.attr("__file__") = file;
  m.attr("m") = m;
  py::eval_file(file);

  return 0;
}
