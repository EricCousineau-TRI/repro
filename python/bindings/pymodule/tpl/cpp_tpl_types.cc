#include "python/bindings/pymodule/tpl/cpp_tpl_types.h"

#include <pybind11/eval.h>

#include "cpp/type_pack.h"
#include "cpp/name_trait.h"


const char kModule[] = "pymodule.tpl.cpp_tpl_types";

TypeRegistry::TypeRegistry() {
  // Import modules into `locals_`.
  globals_ = py::module::import(kModule).attr("__dict__");
  py_to_py_canonical_ = eval("_StrictMap")();

  RegisterCommon();
}

const TypeRegistry& TypeRegistry::GetPyInstance() {
  auto tr_module = py::module::import(kModule);
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
  // Assume this is a Python type.
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
    py_to_py_canonical_.attr("add")(py_type, py_canonical);
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

class TypeRegistry::LiteralHelper {
 public:
  LiteralHelper(TypeRegistry* type_registry)
    : type_registry_(type_registry) {}

  void RegisterLiterals() {
    RegisterSequence(Render(std::integer_sequence<bool, false, true>{}));
    // Register `int` (and `uint` as an alias for positive values).
    constexpr int i_max = 100;
    RegisterSequence(MakeSequence<int, -i_max, -1>());
    RegisterSequence(
        MakeSequence<int, 0, i_max>(),
        {MakeSequence<uint, 0, i_max, int>()});
  }

 private:
  template <typename T>
  struct Sequence {
    std::vector<size_t> keys;
    std::vector<T> values;
  };

  template <typename T, typename Cast = T, T... Values>
  Sequence<Cast> Render(std::integer_sequence<T, Values...>) {
    return Sequence<Cast>{
      {hash_of<std::integral_constant<T, Values>>()...},
      {Values...}};
  }

  template <typename T, T Start, T End, typename Cast = T>
  Sequence<Cast> MakeSequence() {
    constexpr T Count = End - Start + 1;
    auto seq = transform(
        constant_add<T, Start>{}, std::make_integer_sequence<T, Count>{});
    return Render<T, Cast>(seq);
  }

  template <typename T>
  void RegisterSequence(
      const Sequence<T>& seq,
      std::vector<Sequence<T>> alias_set = {}) {
    for (int i = 0; i < seq.keys.size(); ++i) {
      // Get alias types.
      std::vector<size_t> cpp_keys{seq.keys[i]};
      for (const auto& alias : alias_set) {
        assert(seq.values[i] == alias.values[i]);
        cpp_keys.push_back(alias.keys[i]);
      }
      // Register.
      T value = seq.values[i];
      py::object py_value = py::cast(value);
      type_registry_->Register(
          cpp_keys,
          py::make_tuple(py_value),
          py::str(py_value).cast<std::string>());
    }
  }

  TypeRegistry* type_registry_;
};

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
  // Register a subset of literals.
  LiteralHelper(this).RegisterLiterals();
}

py::object TypeRegistry::eval(const std::string& expr) const {
  return py::eval(expr, globals_, locals_);
}

void TypeRegistry::exec(const std::string& expr) {
  py::exec(expr, globals_, locals_);
}
