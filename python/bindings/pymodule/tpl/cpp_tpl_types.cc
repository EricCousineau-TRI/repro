#include "python/bindings/pymodule/tpl/cpp_tpl_types.h"

#include <pybind11/eval.h>

#include "cpp/type_pack.h"
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
    if (!info) {
      throw std::runtime_error("Unknown type!");
    }
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

void TypeRegistry::Register(
      const std::vector<size_t>& cpp_keys,
      py::tuple py_types, const std::string& name) {
  py::handle py_canonical = py_types[0];
  for (size_t cpp_key : cpp_keys) {
    assert(cpp_to_py_.find(cpp_key) == cpp_to_py_.end());
    cpp_to_py_[cpp_key] = py_canonical;
  }
  for (auto py_type : py_types) {
    assert(py_to_py_canonical_.attr("get")(py_type).is_none());
    py_to_py_canonical_[py_type] = py_canonical;
  }
  py_name_[py_canonical] = name;
}

template <typename T>
constexpr size_t hash_of() {
  return std::type_index(typeid(T)).hash_code();
}

template <typename T>
void TypeRegistry::RegisterType(
    py::tuple py_types, const std::string& name_override) {
  std::string name = name_override;
  if (name.empty()) {
    name = py::cast<std::string>(eval("_get_type_name")(py_types[0]));
  }
  Register({hash_of<T>()}, py_types, name);
}

struct TypeRegistry::Helper {
  TypeRegistry* self{};

  using dummy_list = int[];

  void RegisterValue(
      const std::vector<size_t>& cpp_key,
      const std::string& value) {
    std::cout << "register: " << value << std::endl;
    self->Register(
        cpp_key, py::make_tuple(py::eval(value)), value);
  }

  template <typename T, T... Values>
  void RegisterSequence(std::integer_sequence<T, Values...>) {
    (void) dummy_list{(
        RegisterValue(
            {hash_of<std::integral_constant<T, Values>>()},
            std::to_string(Values)),
        0)...};
  }

  template <typename T, T... Values, typename U, U... UValues>
  void RegisterSequenceWithAlias(
      std::integer_sequence<T, Values...>,
      std::integer_sequence<U, UValues...>) {
    (void) dummy_list{(
        assert(Values == T{UValues}),
        RegisterValue(
            {
              hash_of<std::integral_constant<T, Values>>(),
              hash_of<std::integral_constant<U, UValues>>()
            },
            std::to_string(Values)),
        0)...};
  }
};

template <typename T, T Value>
using seq = std::make_integer_sequence<T, Value>;

template <typename T, T Start, T End>
auto make_seq() {
  constexpr T N = End - Start + 1;
  return transform(constant_add<T, Start>{}, seq<T, N>{});
}

void TypeRegistry::RegisterCommon() {
  // Make mappings for C++ RTTI to Python types.
  // Unfortunately, this is hard to obtain from `pybind11`.
  RegisterType<bool>(eval("bool,"));
  RegisterType<std::string>(eval("str,"));
  RegisterType<double>(eval("float, np.double, ctypes.c_double"));
  RegisterType<float>(eval("np.float32, ctypes.c_float"));
  RegisterType<int>(eval("int, np.int32, ctypes.c_int32"));
  RegisterType<uint32_t>(eval("np.uint32, ctypes.c_uint32"));
  RegisterType<int64_t>(eval("np.int64, ctypes.c_int64"));

  Helper h{this};
  h.RegisterSequence(std::integer_sequence<bool, false, true>{});
  constexpr int i_max = 100;
  h.RegisterSequence(make_seq<int, -i_max, -1>());
  h.RegisterSequenceWithAlias(
      make_seq<int, 0, i_max>(),
      make_seq<size_t, 0, i_max>());
}

py::object TypeRegistry::eval(const std::string& expr) const {
  return py::eval(expr, globals_, locals_);
}

void TypeRegistry::exec(const std::string& expr) {
  py::exec(expr, globals_, locals_);
}
