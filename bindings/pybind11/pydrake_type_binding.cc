#include <cstddef>
#include <cmath>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "py_relax.h"

namespace py = pybind11;
using std::string;
using std::ostringstream;

using Eigen::MatrixXd;


class SimpleType {
 public:
  SimpleType() = default;
  SimpleType(const int& value)
    : value_(value) {}
  int value() const { return value_; }
  void set_value(const int& value) { value_ = value; }
 private:
  int value_ {};
};


class EigenType {
 public:
  EigenType() = default;
  EigenType(const MatrixXd& value)
    : value_(value) {}
  const MatrixXd& value() const { return value_; }
  void set_value(const MatrixXd& value) { value_ = value; }
 private:
  MatrixXd value_;
};

PYBIND11_PLUGIN(_pydrake_typebinding) {
  py::module m("_pydrake_typebinding",
               "Drake Type Binding tests");

  // py::object variable = (py::object)
  //   py::module::import("pydrake.symbolic").attr("Variable");

  // TODO: (Learning) Look at py::overload_cast

  // @ref http://pybind11.readthedocs.io/en/master/advanced/functions.html#non-converting-arguments

  py::class_<SimpleType> pySimpleType(m, "SimpleType");
  pySimpleType
    .def(py::init<>())
    .def(py::init<pyint>(), py::arg("value"))
    // .def(py::init<double>()) // Implicit conversion via overload
    .def("value", &SimpleType::value)
    .def("set_value", py_relax_overload<int>(&SimpleType::set_value));
    // .def("set_value", &SimpleType::set_value)
    // // TODO: Make a lambda generator that can emulate pybind11's detail::init<>, or just generate the appropriate thing
    // .def("set_value", [](SimpleType& self, const pyint& value) {
    //   return self.set_value(value);
    // });
    // // Does not work.
    // .def("value", py::overload_cast<double>(&SimpleType::set_value));

  py::class_<EigenType> pyEigenType(m, "EigenType");
  pyEigenType
    .def(py::init<>())
    .def(py::init<MatrixXd>(), py::arg("value"))
    .def("value", &EigenType::value)
    .def("set_value", &EigenType::set_value);

  return m.ptr();
}
