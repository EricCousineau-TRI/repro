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
  return t.__module__ + "." + t.__name__
)""");

  RegisterCommon();
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

py::str TypeRegistry::GetCppName(py::handle py_type) const {
  py::handle py_type_fin = GetPyTypeCanonical(py_type);
  py::object out = py_name_.attr("get")(py_type_fin);
  if (out.is(py::none())) {
    out = eval("_get_type_name")(py_type_fin);
  }
  return out;
}

template <typename T>
void TypeRegistry::Register(
    const std::string& py_values,
    const std::string& cpp_name) {
  size_t cpp_key = std::type_index(typeid(T)).hash_code();
  py::tuple py_types = eval(py_values);
  py::handle py_canonical = py_types[0];
  cpp_to_py_[cpp_key] = py_canonical;
  for (auto t : py_types) {
    py_to_py_canonical_[t] = py_canonical;
  }
  py_name_[py_canonical] = cpp_name.empty() ? nice_type_name<T>() : cpp_name;
}

void TypeRegistry::RegisterCommon() {
  // Make mappings for C++ RTTI to Python types.
  // Unfortunately, this is hard to obtain from `pybind11`.
  Register<bool>("bool,");
  Register<std::string>("str,", "std::string");
  Register<double>("float, np.double, ctypes.c_double");
  Register<float>("np.float32, ctypes.c_float");
  Register<int>("int, np.int32, ctypes.c_int32");
}

py::object TypeRegistry::eval(const std::string& expr) const {
  return py::eval(expr, globals_, locals_);
}

void TypeRegistry::exec(const std::string& expr) {
  py::exec(expr, globals_, locals_);
}
