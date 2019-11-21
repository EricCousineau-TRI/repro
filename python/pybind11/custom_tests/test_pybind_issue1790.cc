// https://github.com/pybind/pybind11/issues/1790

#include <memory>

#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

struct base : public std::enable_shared_from_this<base> {};

struct derived : base {};

void init_module(py::module m) {
  py::class_<base, std::shared_ptr<base>>(m, "Base")
    .def(py::init<>());

  py::class_<derived, base, std::shared_ptr<derived>>(m, "Derived")
    .def(py::init<>());
}


int main(int, char**) {  
  // To Python.
  py::scoped_interpreter guard{};
  py::module m("test_module");
  init_module(m);
  py::globals()["m"] = m;

  py::exec(R"""(
assert issubclass(m.Derived, m.Base)
)""");

  return 0;
}
