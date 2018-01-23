#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "cpp/name_trait.h"
#include "cpp/wrap_function.h"

namespace py = pybind11;

// TODO: Ensure that `keep_alive_impl` can handle these proxies!
// Having the proxy be a patient is fine-ish, as it should keep its proxied
// object alive, but having the proxy as a nurse is bad.

struct py_ref_base : public py::object {};

template <typename T>
struct py_const_ref : public py_ref_base {};

template <typename T>
struct py_mutable_ref : public py_ref_base {};

namespace pybind11 {
namespace detail {

// No check needed if references is constant.
template <typename T>
struct type_caster<py_const_ref<T>> : public type_caster<object> {
  // TODO: Add information for signatures to be user-friendly.
  bool load(handle src, bool convert) {
    value_ = reinterpret_borrow<object>(src);
  }

  static handle cast(py_const_ref<T> src) {
  }

  operator py_const_ref<T>() { return value_; }

 private:
  py_const_ref<T> value_;
};

// If mutable, ensure that input object is not const.
template <typename T>
struct type_caster<py_mutable_ref<T>> : public type_caster<object> {
  // TODO: Add information for signatures to be user-friendly.
};

}  // namespace detail
}  // namespace pybind11

// Wrapping mechanism.

// Checks if a type can actually be tied to an existing reference.
template <typename T>
using is_ref_castable =
    std::is_base<py::detail::type_caster_base<T>, py::detail::type_caster<T>>;

template <typename T, typename = void>
struct wrap_ref<T> : public wrap_arg_default<T> {};

template <typename T>
struct wrap_ref<const T&, std::enable_if_t<is_ref_castable<const T&>::value>> {
  static py_const_ref<T> wrap(const T& arg) {
    return py_const_ref<T>(py::cast(arg));
  }

  static const T& unwrap(py_const_ref<T> arg) {
    return py::cast<const T&>(to_mutable(arg));
  }
};
