// Purpose: Base what avenues might be possible for creating instances in Python
// to then be owned in C++.

#include <cstddef>
#include <cmath>
#include <sstream>
#include <string>

#include <pybind11/cast.h>
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace py::literals;
using namespace std;

struct A {};
struct B {};

int func(unique_ptr<A>) {
  return 0;
}

int func(unique_ptr<B>) {
  return 1;
}

int main() {
  {
    py::scoped_interpreter guard{};

    py::module m("__main__");
    m
      .def("func", py::overload_cast<unique_ptr<A>>(&func))
      .def("func", py::overload_cast<unique_ptr<B>>(&func));

    py::class_<A>(m, "A")
      .def(py::init<>());
    py::class_<B>(m, "B")
      .def(py::init<>());

    py::exec(R"""(
a = A()
b = B()
print(a, b)

assert func(a) == 0
assert func(b) == 1
)""");
  }

  cout << "[ Done ]" << endl;

  return 0;
}
