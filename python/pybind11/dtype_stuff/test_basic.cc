#include <cmath>

#include <map>
#include <iostream>
#include <experimental/optional>

using std::pow;
using std::cerr;
using std::cout;
using std::endl;
using std::experimental::optional;
using std::experimental::nullopt;

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/operators.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>
#include <numpy/ndarraytypes.h>

#include "dtype_custom.h"
#include "ufunc_utility.h"
#include "ufunc_op.h"

namespace py = pybind11;

class Custom {
public:
    Custom() {}
    Custom(double value) : value_{value} {}
    Custom(const Custom&) = default;
    Custom& operator=(const Custom&) = default;
    double value() const { return value_; }

    operator double() const { return value_; }

    Custom operator==(const Custom& rhs) const { return value_ + rhs.value_; }
    Custom operator<(const Custom& rhs) const { return value_ * 10 * rhs.value_; }
    Custom operator*(const Custom& rhs) const {
        return value_ * rhs.value_;
    }
    Custom operator-() const { return -value_; }

    Custom& operator+=(const Custom& rhs) { value_ += rhs.value_; return *this; }
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
    dtype_class<Custom> py_type(m, "Custom");
    // Do not define `__init__` since `cpp_function` is special-purposed for
    // it. Rather, use a custom thing.
    py_type
        .def_dtype(dtype_init<double>())
        // .def(py::self == Class{})
//         // .def(py::self * Class{})
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
        });
  }

  using Class = Custom;
  using Unary = type_pack<Class>;
  using Binary = type_pack<Class, Class>;
  // Arithmetic.
  maybe_ufunc<check_add, Class>(Binary{});
  maybe_ufunc<check_negative, Class>(Unary{});
  maybe_ufunc<check_multiply, Class>(Binary{});
  maybe_ufunc<check_divide, Class>(Binary{});
  maybe_ufunc<check_power, Class>(Binary{});
  maybe_ufunc<check_subtract, Class>(Binary{});
  // Comparison.
  maybe_ufunc<check_greater, Class>(Binary{});
  maybe_ufunc<check_greater_equal, Class>(Binary{});
  maybe_ufunc<check_less, Class>(Binary{});
  maybe_ufunc<check_less_equal, Class>(Binary{});
  maybe_ufunc<check_equal, Class>(Binary{});
  maybe_ufunc<check_not_equal, Class>(Binary{});

  // Casting.
  maybe_cast<Class, double>();
  maybe_cast<double, Class>();
  add_cast<Class, py::object>([](const Class& obj) {
    return py::cast(obj);
  });
  add_cast<py::object, Class>([](py::object obj) {
    return py::cast<Class>(obj);
  });
  // - _zerofill? What do I need?
  add_cast<int, Class>([](int x) {
    return Class{static_cast<double>(x)};
  });
  add_cast<long, Class>([](long x) {
    return Class{static_cast<double>(x)};
  });

  py::str file = "python/pybind11/dtype_stuff/test_basic.py";
  py::print(file);
  m.attr("__file__") = file;
  py::eval_file(file);

  return 0;
}
