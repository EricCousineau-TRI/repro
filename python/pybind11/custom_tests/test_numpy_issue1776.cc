// https://github.com/pybind/pybind11/issues/1776
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "half.h"

namespace py = pybind11;

using float16 = half_float::half;
static_assert(sizeof(float16) == 2, "Bad size");

namespace pybind11 { namespace detail {

}}  // namespace pybind11::detail

void init_module(py::module m) {
  m.def("make_array", []() {
    py::array_t<float16> x({1});
    x.mutable_at(0) = float16{1.};
    return x;
  });
}

int main(int, char**) {
  py::scoped_interpreter guard{};

  py::module m("test_module");
  init_module(m);
  py::globals()["m"] = m;

  py::print("[ Eval ]");
  py::exec(R"""(
import numpy as np

def info(x):
    print(repr(x), type(x))

def main():
    info(m.make_array())

main()
)""");

  py::print("[ Done ]");

  return 0;
}

/* Output:

*/
