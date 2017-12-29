#pragma once

// Purpose: Base what avenues might be possible for creating instances in Python
// to then be owned in C++.

#include <string>
#include <map>

#include <pybind11/pybind11.h>

namespace py = pybind11;

class TypeRegistry {
 public:
  TypeRegistry();

  template <typename T>
  py::handle GetPyType() const {
    return DoGetPyType(typeid(T));
  }

  py::handle GetPyTypeCanonical(py::handle py_type) const;

  py::str GetCppName(py::handle py_type) const;

 private:
  py::handle DoGetPyType(const std::type_info& tinfo) const;

  template <typename T>
  void Register(const std::string& py_values,
                const std::string& cpp_name = "");

  void RegisterCommon();

  py::object eval(const std::string& expr) const;
  void exec(const std::string& expr);

  py::object globals_;
  py::object locals_;
  std::map<size_t, py::handle> cpp_to_py_;
  py::dict py_to_py_canonical_;
  py::dict py_name_;
};
