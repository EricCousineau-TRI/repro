#include <cstddef>
#include <cmath>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "util/py_relax.h"

namespace py = pybind11;
using std::string;
using std::ostringstream;

using Eigen::MatrixXd;

/*
 * Class which requires an integer for construction.
 */
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

/*
 * Class which requires a Matrix for construction.
 */
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

PYBIND11_MODULE(_type_binding, m) {
  m.doc() = "Drake Type Binding tests";

  // Use py_relax_* to make Python bindings more permissive with MATLAB types.

  py::class_<SimpleType> pySimpleType(m, "SimpleType");
  pySimpleType
    .def(py::init<>())
    .def(py_relax_init<int>(), py::arg("value"))
    .def("value", &SimpleType::value)
    .def("set_value", py_relax_overload_cast<int>(&SimpleType::set_value));

  py::class_<EigenType> pyEigenType(m, "EigenType");
  pyEigenType
    .def(py::init<>())
    .def(py_relax_init<MatrixXd>(), py::arg("value"))
    .def("value", &EigenType::value)
    .def("set_value", py_relax_overload_cast<MatrixXd>(&EigenType::set_value));
}
