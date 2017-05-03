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

  // @ref http://pybind11.readthedocs.io/en/master/advanced/functions.html#non-converting-arguments

  py::implicitly_convertible<double, int>();

  py::class_<SimpleType> pySimpleType(m, "SimpleType");
  pySimpleType
    .def(py::init<>())
    .def(py::init<int>(), py::arg("value"))
    // .def(py::init<double>()) // Implicit conversion via overload
    .def("value", &SimpleType::value)
    .def("set_value", &SimpleType::set_value)
    .def("set_value", [](SimpleType& self, double value) {
      self.set_value(value);
    });
    // // Does not work.
    // .def("value", py::overload_cast<double>(&SimpleType::set_value));

  return m.ptr();
}
