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
    py::print("wooh");
  }
  Custom(int value) : value_(value) {}

  int value_{};
};

int main(int argc, char* argv[]) {
    py::scoped_interpreter guard{};
    py::module m("copy_ctor");

    bind_ConstructorStats(m);

    py::class_<Custom>(m, "Custom")
        .def(py::init<Custom>())
        .def(py::init<int>());

    py::dict globals = py::globals();
    globals["m"] = m;

    // py::str file = ;
    // py::print(file);
    // py::eval_file(file);
    py::exec(R"""(
import imp

def _main_impl():
    m = imp.load_source(
        "test_copy_ctor",
        "python/pybind11/custom_tests/test_copy_ctor.py")
    m.main()

import trace, sys
sys.stdout = sys.stderr
tracer = trace.Trace(ignoredirs=sys.path, trace=1, count=0)
print(tracer)
tracer.run('_main_impl()')
)""");

    cout << "[ Done ]" << endl;

    return 0;
}
