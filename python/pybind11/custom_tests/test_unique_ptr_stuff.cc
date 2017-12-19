// Purpose: Test what avenues might be possible for creating instances in Python
// to then be owned in C++.

#include <cstddef>
#include <cmath>
#include <sstream>
#include <string>

#define PYBIND11_WARN_DANGLING_UNIQUE_PYREF

#include <pybind11/cast.h>
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace py::literals;
using namespace std;

int main() {
  py::scoped_interpreter guard{};
  py::module m("test_unique_ptr_stuff");

  class UniquePtrHeld {};
  py::class_<UniquePtrHeld>(m, "UniquePtrHeld")
      .def(py::init<>());

  m.def("unique_ptr_pass_through",
      [](std::unique_ptr<UniquePtrHeld> obj) {
          return obj;
      });

  py::dict globals = py::globals();
  globals["m"] = m;

  py::exec(R"""(
obj = m.UniquePtrHeld()
obj_ref = m.unique_ptr_pass_through(obj)
print(obj_ref)
)""");

  cout << "[ Done ]" << endl;

  return 0;
}
