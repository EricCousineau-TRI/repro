#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "cpp/name_trait.h"
#include "cpp/type_pack.h"
#include "python/bindings/pymodule/tpl/cpp_types.h"

// TODO: Figure out how to handle literals...

namespace py = pybind11;

template <typename T>
inline py::tuple get_py_type(type_pack<T> = {}) {
  return TypeRegistry::GetPyInstance().GetPyType<T>();
}

template <typename ... Ts>
inline py::tuple get_py_types(type_pack<Ts...> = {}) {
  return TypeRegistry::GetPyInstance().GetPyTypes<Ts...>();
}

template <typename T>
inline std::string get_py_name(type_pack<T> = {}) {
  return TypeRegistry::GetPyInstance().GetName<T>();
}

template <typename ... Ts>
inline std::vector<std::string> get_py_names(type_pack<Ts...> = {}) {
  return TypeRegistry::GetPyInstance().GetNames<Ts...>();
}

// Helper class to erase the number of templates needed downstream.
class py_type_pack : public py::tuple {
 public:
  using py::tuple::tuple;

  template <typename ... Ts>
  py_type_pack(type_pack<Ts...> param)
    : py::tuple(get_py_types(param)) {}
};

// Add property / descriptor if it does not already exist.
// This works for adding to classes since `TemplateMethod` acts as a
// descriptor.
py::object InitOrGetTemplate(
    py::handle scope, const std::string& name,
    const std::string& template_type, py::tuple create_extra = py::tuple()) {
  const char module_name[] = "pymodule.tpl.cpp_template";
  py::handle m = py::module::import(module_name);
  return m.attr("init_or_get")(
      scope, name, m.attr(template_type.c_str()), *create_extra);
}

void AddInstantiation(
    py::handle tpl, py::handle obj,
    py_type_pack param) {
  tpl.attr("add_instantiation")(param, obj);
}

py::object AddTemplateClass(
    py::handle scope, const std::string& name,
    py::handle py_class,
    py_type_pack param,
    const std::string& default_inst_name = "") {
  py::object tpl = InitOrGetTemplate(scope, name, "TemplateClass");
  AddInstantiation(tpl, py_class, param);
  if (!default_inst_name.empty() &&
      !py::hasattr(scope, default_inst_name.c_str())) {
    scope.attr(default_inst_name.c_str()) = py_class;
  }
  return tpl;
}

template <typename T>
std::string TemplateClassName() {
  return std::string("_TmpTemplate_") + typeid(T).name();
}

std::string GetInstantiationName(py::handle tpl, py_type_pack param) {
  return py::cast<std::string>(
    tpl.attr("_get_instantiation_name")(param));
}

// @note Overloads won't be allowed with templates. If it is needed,
// see `py::sibling(...)`.
template <typename Func>
py::object AddTemplateFunction(
    py::handle scope, const std::string& name, Func&& func,
    py_type_pack param) {
  py::object tpl = InitOrGetTemplate(scope, name, "TemplateFunction");
  py::object py_func = py::cpp_function(
        std::forward<Func>(func),
        py::name(GetInstantiationName(tpl, param).c_str()));
  AddInstantiation(tpl, py_func, param);
  return tpl;
}

template <typename Func>
py::object AddTemplateMethod(
    py::handle scope, const std::string& name, Func&& func,
    py_type_pack param) {
  py::object tpl = InitOrGetTemplate(
      scope, name, "TemplateMethod", py::make_tuple(scope));
  py::object py_func = py::cpp_function(
        std::forward<Func>(func),
        py::name(GetInstantiationName(tpl, param).c_str()),
        py::is_method(scope));
  AddInstantiation(tpl, py_func, param);
  return tpl;
}
