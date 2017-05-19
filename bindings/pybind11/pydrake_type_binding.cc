#include <cstddef>
#include <cmath>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

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

// How to swap out types? Meh. Only accept const-value types???
template <typename T>
struct py_relax_type { using type = T; };

template <>
struct py_relax_type<int> { using type = pyint; };

// Does std:: have this?
  // Can't find it here
  // http://en.cppreference.com/w/cpp/header/functional
template <class R, class T, class... Args>
T mem_fn_class_impl(R (T::* pm)(Args...));

template <typename Method>
struct mem_fn_class {
  using type = decltype(mem_fn_class_impl(std::declval<Method>()));
};

template <typename ... Args, typename Method>
auto py_relax_overload(const Method& method) {
  using Base = typename mem_fn_class<Method>::type;
  auto relaxed = [=](
      Base* self, const typename py_relax_type<Args>::type&... args) {
    return (self->*method)(args...);
  };
  return relaxed;
}

namespace pybind11 {
namespace detail {

// Helper conversion methods
template<typename T>
struct py_conversion { };

#define PYBIND11_PY_CONVERSION(T, PyAsT, PyFromT) \
  template<> struct py_conversion<T> { \
    static T from_py(handle src) { return PyAsT(src.ptr()); } \
    static handle to_py(T src) { return PyFromT(src); } \
  }

PYBIND11_PY_CONVERSION(int, PyLong_AsLong, PyLong_FromLong);

// Following:
// pybind11/include/pybind11/cast.h
//   struct type_caster<T, enable_if_t<std::is_arithmetic<T>::value &&  //...

template <typename T>
struct type_caster<PyIntegral<T>> {
  PYBIND11_TYPE_CASTER(pyint, _("pyint"));

  bool load(handle src, bool convert) {
    if (PyFloat_Check(src.ptr())) {
      double tmp = PyFloat_AsDouble(src.ptr());
      if (tmp == -1. && PyErr_Occurred()) {
        return false;
      }
      value = tmp;
    } else {
      T tmp = py_conversion<T>::from_py(src);
      // assuming unsigned logic is correct
      if (tmp == (T)-1 && PyErr_Occurred()) {
        return false;
      }
      value = tmp;
    }
    return true;
  }

  static handle cast(pyint src, return_value_policy /* policy */,
                     handle /* parent */) {
    return py_conversion<T>::to_py(src);
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
