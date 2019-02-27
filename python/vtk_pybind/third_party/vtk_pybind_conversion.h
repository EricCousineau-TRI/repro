// File from: https://gitlab.kitware.com/cmb/smtk/blob/9bf5b4f9/smtk/extension/vtk/pybind11/PybindVTKTypeCaster.h
// Except for portions otherwise denoted.

#pragma once

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>

#include <type_traits>

#include <vtkObjectBase.h>
#include <vtkPythonUtil.h>
#include <vtkSmartPointer.h>

// From: https://stackoverflow.com/q/54871216/7829525
PYBIND11_DECLARE_HOLDER_TYPE(T, vtkSmartPointer<T>);

namespace pybind11 {
namespace detail {

template <typename Class>
struct type_caster<
    Class,
    enable_if_t<std::is_base_of<vtkObjectBase, Class>::value>
    > {
 protected:
  Class* value;
 public:
  static constexpr auto name = _<Class>();
  template <typename T_,
            enable_if_t<std::is_same<Class,
                                     remove_cv_t<T_>>::value, int> = 0>
  static handle cast(T_ *src, return_value_policy policy, handle parent)
   {
    if (!src) return none().release();
    if (policy == return_value_policy::take_ownership) {
      auto h = cast(std::move(*src), policy, parent);
      return h;
    } else {
      return cast(*src, policy, parent);
    }
  }
  operator Class*() { return value; }
  operator Class&() { return *value; }
  operator Class&&() && { return std::move(*value); }
  template <typename T_>
  using cast_op_type = pybind11::detail::movable_cast_op_type<T_>;
  bool load(handle src, bool) {
    value = dynamic_cast<Class*>(
      vtkPythonUtil::GetPointerFromObject(src.ptr(), type_id<Class>().c_str()));
    if (!value) {
      PyErr_Clear();
      throw reference_cast_error();
    }
    return value != nullptr;
  }
  static handle cast(const Class& src, return_value_policy, handle) {
    return vtkPythonUtil::GetObjectFromPointer(
      const_cast<Class*>(&src));
  }
};

}  // namespace detail
}  // namespace pybind11
