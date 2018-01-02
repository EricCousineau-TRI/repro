#include "python/bindings/pymodule/tpl/cpp_tpl_types.h"

#include <pybind11/eval.h>

#include "cpp/name_trait.h"


TypeRegistry::TypeRegistry() {
  // Import modules into `locals_`.
  globals_ = py::globals();
  exec(R"""(
import numpy as np; import ctypes

def _get_type_name(t):
  # Gets scoped type name as a string.
  prefix = t.__module__ + "."
  if prefix == "__builtin__.":
    prefix = ""
  return prefix + t.__name__
)""");

  RegisterCommon();
}

const TypeRegistry& TypeRegistry::GetPyInstance() {
  auto tr_module = py::module::import("pymodule.tpl.cpp_tpl_types");
  py::object type_registry_py = tr_module.attr("type_registry");
  const TypeRegistry* type_registry =
      py::cast<const TypeRegistry*>(type_registry_py);
  return *type_registry;
}

py::handle TypeRegistry::DoGetPyType(const std::type_info& tinfo) const {
  // Check if it's a custom-registered type.
  size_t cpp_key = std::type_index(tinfo).hash_code();
  auto iter = cpp_to_py_.find(cpp_key);
  if (iter != cpp_to_py_.end()) {
    return iter->second;
  } else {
    // Get from pybind11-registered types.
    // WARNING: Internal API :(
    auto* info = py::detail::get_type_info(tinfo);
    assert(info != nullptr);
    return py::handle(reinterpret_cast<PyObject*>(info->type));
  }
}

py::handle TypeRegistry::GetPyTypeCanonical(py::handle py_type) const {
  // Since there's no easy / good way to expose C++ type id's to Python,
  // just canonicalize Python types.
  return py_to_py_canonical_.attr("get")(py_type, py_type);
}

py::tuple TypeRegistry::GetPyTypesCanonical(py::tuple py_types) const {
  py::tuple out(py_types.size());
  for (int i = 0; i < py_types.size(); ++i) {
    out[i] = GetPyTypeCanonical(py_types[i]);
  }
  return out;
}

py::str TypeRegistry::GetName(py::handle py_type) const {
  py::handle py_type_fin = GetPyTypeCanonical(py_type);
  py::object out = py_name_.attr("get")(py_type_fin);
  if (out.is(py::none())) {
    out = eval("_get_type_name")(py_type_fin);
  }
  return out;
}

py::tuple TypeRegistry::GetNames(py::tuple py_types) const {
  py::tuple out(py_types.size());
  for (int i = 0; i < py_types.size(); ++i) {
    out[i] = GetName(py_types[i]);
  }
  return out;
}

template <typename T>
void TypeRegistry::Register(
    py::tuple py_types, const std::string& name_override) {
  py::handle py_canonical = py_types[0];
  size_t cpp_key = std::type_index(typeid(T)).hash_code();
  cpp_to_py_[cpp_key] = py_canonical;
  for (auto py_type : py_types) {
    py_to_py_canonical_[py_type] = py_canonical;
  }
  if (!name_override.empty()) {
    py_name_[py_canonical] = name_override;
  } else {
    py_name_[py_canonical] =
        py::cast<std::string>(eval("_get_type_name")(py_canonical));
  }
}

void TypeRegistry::RegisterCommon() {
  // Make mappings for C++ RTTI to Python types.
  // Unfortunately, this is hard to obtain from `pybind11`.
  Register<bool>(eval("bool,"));
  Register<std::string>(eval("str,"));
  Register<double>(eval("float, np.double, ctypes.c_double"));
  Register<float>(eval("np.float32, ctypes.c_float"));
  Register<int>(eval("int, np.int32, ctypes.c_int32"));
  Register<uint32_t>(eval("np.uint32, ctypes.c_uint32"));
  Register<int64_t>(eval("np.int64, ctypes.c_int64"));
}

py::object TypeRegistry::eval(const std::string& expr) const {
  return py::eval(expr, globals_, locals_);
}

void TypeRegistry::exec(const std::string& expr) {
  py::exec(expr, globals_, locals_);
}
