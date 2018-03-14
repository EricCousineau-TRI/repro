#include <cmath>

#include <map>
#include <iostream>
#include <experimental/optional>

#include <Eigen/Dense>

using std::pow;
using std::cerr;
using std::cout;
using std::endl;
using std::experimental::optional;
using std::experimental::nullopt;
using Eigen::Ref;

template <typename T>
using MatrixX = Eigen::Matrix<T, -1, -1>;

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/operators.h>
#include <pybind11/eigen.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>
#include <numpy/ndarraytypes.h>

#include "dtype_user.h"
#include "ufunc_utility.h"
#include "ufunc_op.h"

namespace py = pybind11;

class Custom {
public:
    Custom() {
      // cerr << "Construct\n";
    }
    ~Custom() {
      // cerr << "Destruct\n";
    }
    Custom(double value) : value_{value} {
      // cerr << "Construct\n";
    }
    Custom(const Custom& other) {
      // cerr << "Construct\n";
      value_ = other.value_;
    }
    Custom& operator=(const Custom&) = default;
    double value() const { return value_; }

    operator double() const { return value_; }

    Custom operator==(const Custom& rhs) const { return value_ + rhs.value_; }
    Custom operator<(const Custom& rhs) const { return value_ * 10 * rhs.value_; }
    Custom operator*(const Custom& rhs) const {
        return value_ * rhs.value_;
    }
    Custom operator-() const { return -value_; }

    Custom& operator+=(const Custom& rhs) {
      value_ += rhs.value_; return *this;
    }
    Custom operator+(const Custom& rhs) const { return Custom(*this) += rhs.value_; }
    Custom operator-(const Custom& rhs) const { return value_ - rhs.value_; }

private:
    double value_{};
};

Custom pow(Custom a, Custom b) {
  return pow(a.value(), b.value());
}

namespace pybind11 { namespace detail {

template <>
struct type_caster<Custom> : public dtype_caster<Custom> {};
template <>
struct npy_format_descriptor<Custom> : public npy_format_descriptor_custom<Custom> {};

} } // namespace pybind11 { namespace detail {

int main() {
  py::scoped_interpreter guard;

  py::module m("__main__");
  py::dict md = m.attr("__dict__");
  py::dict locals;

  {
    dtype_user<Custom> py_type(m, "Custom");
    // Do not define `__init__` since `cpp_function` is special-purposed for
    // it. Rather, use a custom thing.
    py_type
        .def(py::init<double>())
        .def("value", &Custom::value)
        .def("incr", [](Custom* self) {
          cerr << "incr\n";
          *self += 10;
        })
        .def("__repr__", [](const Custom* self) {
          return py::str("<Custom({})>").format(self->value());
        })
        .def("__str__", [](const Custom* self) {
          return py::str("Custom({})").format(self->value());
        })
        .def("self", [](Custom* self) {
          return self;
        }, py::return_value_policy::reference)
        // Operators + ufuncs, with some just-operators (e.g. in-place)
        .def_ufunc(py::self + py::self)
        .def(py::self += py::self)
        .def_ufunc(-py::self)
        .def_ufunc(py::self == py::self)
        .def_ufunc(py::self < py::self)
        .def_ufunc(py::self * py::self)
        .def_ufunc(py::self + py::self)
        .def_ufunc_cast([](const Custom& in) -> double { return in; })
        .def_ufunc_cast([](const double& in) -> Custom { return in; });
  }

  // Define a mutating function.
  m.def("mutate", [](Ref<MatrixX<Custom>> value) {
    value.array() += Custom(100);
  });

  py::str file = "python/pybind11/dtype_stuff/test_basic.py";
  py::print(file);
  m.attr("__file__") = file;
  py::eval_file(file);

  return 0;
}
