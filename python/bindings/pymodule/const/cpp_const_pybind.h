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

// Opaquely wrap `py::object` (such that it does not confused `is_pyobject<>`)
// so that we can distinguish situations.
template <typename T>
struct py_const_ref {
  py::object obj;
};

template <typename T>
struct py_mutable_ref {
  py::object obj;
};

inline bool is_const(py::handle h) {
  py::module m = py::module::import("cpp_const");
  return m.attr("is_const_or_immutable")(h).cast<bool>();
}

inline py::object to_mutable(py::handle h, bool force = false) {
  py::module m = py::module::import("cpp_const");
  return m.attr("to_mutable")(h, force);
}

inline py::object to_const(py::handle h) {
  py::module m = py::module::import("cpp_const");
  return m.attr("to_const")(h);
}

namespace pybind11 {
namespace detail {

// No check needed if references is constant.
template <typename T>
struct type_caster<py_const_ref<T>> : public type_caster<object> {
  // TODO: Add information for type name to be user-friendly.
  PYBIND11_TYPE_CASTER(py_const_ref<T>, _("py_const_ref<T>"));

  bool load(handle src, bool convert) {
    value.obj = to_mutable(src, true);
    return true;
  }

  static handle cast(py_const_ref<T> src, return_value_policy, handle) {
    // Ensure Python object is const-proxied.
    // TODO: Somehow intercept keep alive behavior here?
    object obj = to_const(src.obj);
    return obj.release();  // Uh... ???
  }
};

// If mutable, ensure that input object is not const.
template <typename T>
struct type_caster<py_mutable_ref<T>> : public type_caster<object> {
  PYBIND11_TYPE_CASTER(py_mutable_ref<T>, _("py_mutable_ref<T>"));

  bool load(handle src, bool convert) {
    if (is_const(src)) {
      // Do not allow loading const-proxied values.
      return false;
    } else {
      value.obj = to_mutable(src);
      return true;
    }
  }

  static handle cast(py_mutable_ref<T> src, return_value_policy, handle) {
    return src.obj.release();  // Uh... ???
  }

 private:
  py_mutable_ref<T> value_;
};

}  // namespace detail
}  // namespace pybind11

// Wrapping mechanism.

// Checks if a type can actually be tied to an existing reference.
template <typename T>
using is_ref_castable =
    std::is_base_of<
        py::detail::type_caster_base<T>,
        py::detail::type_caster<T>>;

template <typename T, typename = void>
struct wrap_ref : public wrap_arg_default<T> {};

template <typename T>
using is_ref_or_ptr =
    std::integral_constant<bool,
        std::is_reference<T>::value || std::is_pointer<T>::value>;

template <typename T>
using remove_ref_or_ptr_t =
    typename std::remove_reference<
        typename std::remove_pointer<T>::type>::type;

template <typename T>
using is_const_ref_or_ptr =
    std::integral_constant<bool,
        is_ref_or_ptr<T>::value &&
        std::is_const<remove_ref_or_ptr_t<T>>::value>;

// TODO: Add check for `unique_ptr<{T, const T}>`, `shared_ptr<{T, const T}>`.

template <typename T, bool is_const = true>
struct wrap_ref_type {
  using type = py_const_ref<T>;
};

template <typename T>
struct wrap_ref_type<T, false> {
  using type = py_mutable_ref<T>;
};

template <typename T>
using wrap_ref_t =
    typename wrap_ref_type<
        std::decay_t<T>, is_const_ref_or_ptr<T>::value>::type;

// This is effectively done to augment pybind's existing specializations for
// type_caster<U>, where U is all of {T*, T&, const T*, const T&}
template <typename T>
struct wrap_ref<T, std::enable_if_t<is_ref_castable<T>::value>> {
  static wrap_ref_t<T> wrap(T arg) {
    return {py::cast(arg)};
  }

  static T unwrap(wrap_ref_t<T> arg_wrapped) {
    return py::cast<T>(arg_wrapped.obj);
  }
};

template <typename Func>
auto WrapRef(Func&& func) {
  return WrapFunction<wrap_ref>(std::forward<Func>(func));
}
