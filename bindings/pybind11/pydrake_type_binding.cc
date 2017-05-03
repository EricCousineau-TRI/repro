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

  py::class_<SimpleType> pySimpleType(m, "SimpleType");
  pySimpleType
    .def("value", &SimpleType::value)
    .def("set_value", &SimpleType::set_value);

  return m.ptr();
}
