// Purpose: Test to see if we can bind methods using `py::object self`
// (to then help pave the way for exposing `const` properties).

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

class Test {
 public:
  Test(int value) : value_(value) {}

  int get_value() const { return value_; }
  void set_value(int value) { value_ = value; }
 private:
  int value_{};
};

int main(int argc, char* argv[]) {
    py::scoped_interpreter guard{};
    py::module m("_test_const");

    // bind_ConstructorStats(m);

    py::class_<Test>(m, "Test")
        .def(py::init<int>())
        .def("get_value", [](py::object self) {
          // Const, no need to check.
          return self.cast<const Test*>()->get_value();
        })
        .def("set_value", [](py::object self, int value) {
          // Mutable, should check that input is non-const.
          return self.cast<Test*>()->set_value(value);
        });

    py::dict globals = py::globals();
    globals["m"] = m;

    py::str file;
    if (argc < 2) {
        file = "python/pybind11/custom_tests/test_const.py";
    } else {
        file = argv[1];
    }
    py::print(file);
    py::eval_file(file);

    cout << "[ Done ]" << endl;

    return 0;
}
