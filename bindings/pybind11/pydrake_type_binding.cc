#include <cstddef>
#include <cmath>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using std::string;
using std::ostringstream;


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

// Flexible integer with safety checking
template <typename T = int>
class PyIntegral {
public:
  // Need default ctor to permit type_caster to be constructible per macro
  PyIntegral() = default; 
  PyIntegral(T value)
    : value_(value) {}
  PyIntegral(double value) {
    *this = value;
  }
  operator T() const { return value_; }
  PyIntegral& operator=(T value) {
    value_ = value;
    return *this;
  }
  PyIntegral& operator=(const PyIntegral& other) = default;
  PyIntegral& operator=(const double& value) {
    T tmp = static_cast<T>(value);
    // Ensure they are approximately the same
    double err = value - tmp;
    if (std::abs(err) > 1e-8) {
      ostringstream msg;
      msg << "Only integer-valued floating point values accepted"
          << " (input: " << value << ", integral distance: " << err << ")";
      throw std::runtime_error(msg.str());
    }
    return *this = tmp;
  }
private:
  T value_ {};
};

using pyint = PyIntegral<int>;

namespace pybind11 {
namespace detail {

// Look at;
// pybind11 ... cast.h
// struct type_caster<T, enable_if_t<std::is_arithmetic<T>::value &&  //...

template <>
struct type_caster<pyint> {
  PYBIND11_TYPE_CASTER(pyint, _("pyint"));

  bool load(handle src, bool convert) {
    if (PyFloat_Check(src.ptr())) {
      double tmp = PyFloat_AsDouble(src.ptr());
      if (tmp == -1. && PyErr_Occurred()) {
        return false;
      }
      value = tmp;
    } else {
      int tmp = PyLong_AsLong(src.ptr());
      if (tmp == -1 && PyErr_Occurred()) {
        return false;
      }
      value = tmp;
    }
    return true;
  }

  static handle cast(pyint src, return_value_policy /* policy */,
                     handle /* parent */) {
    return PyLong_FromLong(src);
  }
};


} // namespace pybind11
} // namespace detail



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
    .def("set_value", &SimpleType::set_value)
    .def("set_value", [](SimpleType& self, double value) {
      self.set_value(value);
    });
    // // Does not work.
    // .def("value", py::overload_cast<double>(&SimpleType::set_value));

  return m.ptr();
}
