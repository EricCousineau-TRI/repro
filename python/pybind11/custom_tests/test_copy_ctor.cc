// Purpose: Debug unique ptr casting.

#include <cstddef>
#include <cmath>
#include <sstream>
#include <string>

#include <pybind11/cast.h>
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "constructor_stats.h"

namespace py = pybind11;
using namespace py::literals;
using namespace std;

void bind_ConstructorStats(py::module &m);

struct Custom {
  Custom(const Custom& other)
      : value_(other.value_) {
    py::print("Custom");
  }
  Custom(int value) : value_(value) {
    py::print("int");
  }

  operator int() const {
    py::print("cast");
    return value_;
  }

  int value_{};
};

int main(int argc, char* argv[]) {
    py::scoped_interpreter guard{};
    py::module m("copy_ctor");

    bind_ConstructorStats(m);

    py::class_<Custom>(m, "Custom")
        .def(py::init<int>())
        .def(py::init<Custom>())
        .def("__int__", &Custom::operator int);
    py::implicitly_convertible<int, Custom>();

    // Execute it new set of globals?
    py::exec(R"""(
import imp
imp.load_source(
   "test_copy_ctor",
   "python/pybind11/custom_tests/test_copy_ctor.py")
)""");

    cout << "[ Done ]" << endl;

    return 0;
}
