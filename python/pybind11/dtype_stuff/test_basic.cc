#include <cmath>

#include <map>
#include <iostream>
#include <experimental/optional>

#include <Eigen/Dense>

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/operators.h>
#include <pybind11/eigen.h>
#include "pybind11/functional.h"
#include <pybind11/stl.h>
#include <pybind11/numpy_dtypes_user.h>

using std::pow;
using std::cerr;
using std::cout;
using std::endl;
using std::experimental::optional;
using std::experimental::nullopt;
using Eigen::Ref;

template <typename T>
using MatrixX = Eigen::Matrix<T, -1, -1>;
using Eigen::VectorXd;

namespace py = pybind11;

class Custom {
public:
    Custom() {
      stuff_.resize(3);
      stuff_ << 1, 2, 3;
      // cerr << "Construct\n";
    }
    ~Custom() {
      // cerr << "Destruct\n";
    }
    Custom(double value) : value_{value} {
      stuff_.resize(3);
      stuff_ << 1, 2, 3;
      // cerr << "Construct\n";
    }

    Custom(double value, const VectorXd& stuff)
      : value_(value), stuff_(stuff) {}

    Custom(const Custom& other) {
      // cerr << "Construct\n";
      value_ = other.value_;
      stuff_ = other.stuff_;
    }
    Custom& operator=(const Custom& other) {
      value_ = other.value_;
      stuff_ = other.stuff_;
      return *this;
    }
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

    VectorXd stuff() const { return stuff_; }

private:
    double value_{};
    VectorXd stuff_;
};

Custom pow(Custom a, Custom b) {
  return pow(a.value(), b.value());
}

template <typename T>
using storage_for = typename std::aligned_storage<sizeof(T), alignof(T)>::type;

PYBIND11_NUMPY_DTYPE_USER(Custom);

int main() {
  py::scoped_interpreter guard;

  py::module m("__main__");
  py::dict md = m.attr("__dict__");
  py::dict locals;

  {
    py::dtype_user<Custom> py_type(m, "Custom");
    // Do not define `__init__` since `cpp_function` is special-purposed for
    // it. Rather, use a custom thing.
    py_type
        .def(py::init<double>())
        .def(py::init<double, const VectorXd&>())
        .def("value", &Custom::value)
        .def("incr", [](Custom* self) {
          cerr << "incr\n";
          *self += 10;
        })
        .def("__repr__", [](const Custom* self) {
          return py::str("<Custom({} - {})>").format(self->value(), self->stuff());
        })
        .def("__str__", [](const Custom* self) {
          return py::str("Custom({} - {})").format(self->value(), self->stuff());
        })
        .def("self", [](Custom* self) {
          return self;
        }, py::return_value_policy::reference)
        // Operators + ufuncs, with some just-operators (e.g. in-place)
        .def_loop(py::self + py::self)
        .def(py::self += py::self)
        .def_loop(-py::self)
        .def_loop(py::self == py::self)
        .def_loop(py::self < py::self)
        .def_loop(py::self * py::self)
        .def_loop(py::self + py::self)
        .def_loop_cast([](const Custom& in) -> double { return in; })
        .def_loop_cast([](const double& in) -> Custom { return in; })
        // For reduction methods identity functions, for NumPy <= 1.14, you must define
        // a cast from `bool` to this type for correct behavior for reduction.
        // This may have been fixed by: https://github.com/numpy/numpy/pull/8952
        .def_loop_cast([](const bool& in) -> Custom { return in; });
  }

  // Define a mutating function.
  m.def("mutate", [](Ref<MatrixX<Custom>> value) {
    value.array() += Custom(100);
  });

  // Test `std::function` stuff.
  m.def("call_func", [](py::function f) {
    using Func = std::function<MatrixX<Custom>(const MatrixX<Custom>&)>;
    auto f_cpp = py::cast<Func>(f);
    MatrixX<Custom> output;
    {
      MatrixX<Custom> value(1, 2);
      value << Custom(1), Custom(2);
      py::print("Call func");
      output = f_cpp(value);
      cerr << "value: " << value.transpose() << endl;
    }
    py::print("output: ", output);
  });

  py::str file = "python/pybind11/dtype_stuff/test_basic.py";
  py::print(file);
  m.attr("__file__") = file;
  py::eval_file(file);

  return 0;
}
