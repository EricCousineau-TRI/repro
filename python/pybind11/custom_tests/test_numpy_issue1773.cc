// From: https://github.com/pybind/pybind11/issues/1773
#include <string>
#include <vector>

#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void init_module(py::module m) {
  m.def("make_string_array", []() {
    std::vector<std::string> values = {
      "hello", "world", "you", "cool", "asdfasdfasdfasdf"};
    return py::array(py::cast(values));
  });
}

int main(int, char**) {
  py::scoped_interpreter guard{};

  py::module m("test_module");
  init_module(m);
  py::globals()["m"] = m;

  py::print("[ Eval ]");
  py::exec(R"""(
print(repr(m.make_string_array()))
)""");

  py::print("[ Done ]");

  return 0;
}
