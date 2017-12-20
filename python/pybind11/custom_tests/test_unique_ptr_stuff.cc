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

  class SharedPtrHeld {
   public:
    SharedPtrHeld(int value)
      : value_(value) {}
    int value() const { return value_; }
   private:
    int value_;
  };
  py::class_<SharedPtrHeld, std::shared_ptr<SharedPtrHeld>>(m, "SharedPtrHeld")
      .def(py::init<int>())
      .def("value", &SharedPtrHeld::value);
  m.def("shared_ptr_held_in_unique_ptr",
      []() {
          return std::make_unique<SharedPtrHeld>(1);
      });
  m.def("shared_ptr_held_func",
      [](std::shared_ptr<SharedPtrHeld> obj) {
          return obj != nullptr;
      });

  py::dict globals = py::globals();
  globals["m"] = m;

  py::exec(R"""(
obj = m.shared_ptr_held_in_unique_ptr()
print(obj.value())
#assert m.shared_ptr_held_func(obj) == True
print(obj)
)""");

  cout << "[ Done ]" << endl;

  return 0;
}
