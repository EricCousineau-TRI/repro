#include <cstddef>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using std::string;


class SimpleType {
 public:
  SimpleType() = default;
  SimpleType(int value)
    : value_(value) {}
  int value() const { return value_; }
  void set_value(int value) { value_ = value; }
 private:
  int value_ {};
};


PYBIND11_PLUGIN(_pydrake_typebinding) {
  py::module m("_pydrake_typebinding",
               "Drake Type Binding tests");

  // py::object variable = (py::object)
  //   py::module::import("pydrake.symbolic").attr("Variable");

  // TODO: (Learning) Look at py::overload_cast

  auto init_double = [](SimpleType& self, double value) {
    new (&self) SimpleType(value);
  };

  py::class_<SimpleType> pySimpleType(m, "SimpleType");
  pySimpleType
    .def(py::init<>())
    .def(py::init<int>())
    .def("__init__", init_double)
    .def("value", &SimpleType::value)
    .def("set_value", &SimpleType::set_value);

  return m.ptr();
}
