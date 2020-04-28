#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

class Example {
 public:
  Example() {}
  Example(int) {}

  void method() {}
  void method(int) {}
};

PYBIND11_MODULE(example, m) {
  py::class_<Example>(m, "Example")
      .def(py::init(), "Init 1")
      .def(py::init<int>(), "Init 2")
      .def("method", py::overload_cast<>(&Example::method), "Method 1")
      .def("method", py::overload_cast<int>(&Example::method), "Method 2");
}

}  // namespace
