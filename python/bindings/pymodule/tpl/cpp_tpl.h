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
  py::handle TemplateMethod =
      py::module::import("pymodule.tpl.cpp_tpl").attr("TemplateMethod");
  std::string tpl_attr = "_tpl_" + name;
  py::object tpl = py::getattr(py_class, tpl_attr.c_str(), py::none());
  using Class = typename PyClass::type;
  if (tpl.is(py::none())) {
    // Add template backend.
    tpl = TemplateMethod(name, py_class);
    py::setattr(py_class, tpl_attr.c_str(), tpl);
    // Add read-only property.
    py_class.def_property(
        name.c_str(),
        [tpl](Class* self) {
          return tpl.attr("bind")(self);
        },
        // TODO: Fix this once pybind is upgraded.
        [](Class* self, py::handle) {
          throw std::runtime_error("Read-only");
        });
  }
  // Ensure that pybind is aware that it's a function.
  auto cpp_instantiation_func = [instantiation_func](auto param) {
    return py::cpp_function(instantiation_func(param));
  };
  RegisterInstantiations(tpl, cpp_instantiation_func, param_list);
  return tpl;
}
