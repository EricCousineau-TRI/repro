#pragma once

// Purpose: Base what avenues might be possible for creating instances in Python
// to then be owned in C++.

// TODO: Figure out how to handle literals...

#include <string>
#include <map>

#include <pybind11/pybind11.h>

namespace py = pybind11;

class TypeRegistry {
 public:
  TypeRegistry();

  static const TypeRegistry& GetPyInstance();

  template <typename T>
  py::handle GetPyType() const {
    return DoGetPyType(typeid(T));
  }

  template <typename ... Ts>
  py::tuple GetPyTypes() const {
    return py::make_tuple(GetPyType<Ts>()...);
  }

  py::handle GetPyTypeCanonical(py::handle py_type) const;

  py::tuple GetPyTypesCanonical(py::tuple py_types) const;

  py::str GetName(py::handle py_type) const;

  py::tuple GetNames(py::tuple py_types) const;

  template <typename T>
  std::string GetName() const {
    return py::cast<std::string>(GetName(GetPyType<T>()));
  }

  template <typename ... Ts>
  std::vector<std::string> GetNames() const {
    return {GetName<Ts>()...};
  }

 private:
  py::handle DoGetPyType(const std::type_info& tinfo) const;

  void Register(
      size_t cpp_key, py::tuple py_types, const std::string& name);

  template <typename T>
  void RegisterType(py::tuple py_types,
                const std::string& name_override = {});

  void RegisterCommon();

  py::object eval(const std::string& expr) const;
  void exec(const std::string& expr);

  py::object globals_;
  py::object locals_;
  std::map<size_t, py::handle> cpp_to_py_;
  py::dict py_to_py_canonical_;
  py::dict py_name_;

  struct Helper;
  friend struct Helper;
};
