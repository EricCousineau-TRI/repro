#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "cpp/name_trait.h"
#include "cpp/type_pack.h"
#include "python/bindings/pymodule/tpl/cpp_tpl_types.h"

// TODO: Figure out how to handle literals...

namespace py = pybind11;

template <typename ... Args>
inline py::tuple get_py_types(type_pack<Args...> = {}) {
  return TypeRegistry::GetPyInstance().GetPyTypes<Args...>();
}

template <typename ParamList, typename InstantiationFunc>
void RegisterInstantiations(
    py::object tpl, const InstantiationFunc& instantiation_func,
    ParamList = {}) {
  auto add_instantiation = [&](auto param) {
    // Register instantiation in `pybind`, using lambda
    // `auto`-friendly syntax., indexing by canonical Python types.
    tpl.attr("add_instantiation")(
        get_py_types(param),
        instantiation_func(param));
  };
  ParamList::template visit<no_tag>(add_instantiation);
}

template <
    typename ParamList, typename PyClass,
    typename InstantiationFunc>
py::object RegisterTemplateMethod(
    PyClass& py_class, const std::string& name,
    const InstantiationFunc& instantiation_func, ParamList param_list = {}) {
  py::handle TemplateFunction =
      py::module::import("pymodule.tpl.cpp_tpl").attr("TemplateFunction");
  // Add property / descriptor if it does not already exist.
  py::object tpl = py::getattr(py_class, name.c_str(), py::none());
  if (tpl.is(py::none())) {
    tpl = TemplateFunction(name, py_class);
    py::setattr(py_class, name.c_str(), tpl);
  }
  // Ensure that pybind is aware that it's a function.
  py::object py_type(py_class);  // Keep capture type simple.
  auto cpp_instantiation_func =
      [instantiation_func, py_type, tpl](auto param) {
    std::string instantiation_name =
        py::cast<std::string>(
            tpl.attr("_get_instantiation_name")(get_py_types(param)));
    return py::cpp_function(
        instantiation_func(param), py::name(instantiation_name.c_str()),
        py::is_method(py_type));
  };
  RegisterInstantiations(tpl, cpp_instantiation_func, param_list);
  return tpl;
}
