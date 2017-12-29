// Purpose: Base what avenues might be possible for creating instances in Python
// to then be owned in C++.

#include <cstddef>
#include <cmath>
#include <sstream>
#include <string>
#include <map>

#include <pybind11/cast.h>
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/pybind11.h>

#include "cpp/name_trait.h"

namespace py = pybind11;
using namespace py::literals;
using namespace std;

class SimpleType {
 public:
    SimpleType(int value)
        : value_(value) {
      cout << "SimpleType::SimpleType()" << endl;
    }
    ~SimpleType() {
      cout << "SimpleType::~SimpleType()" << endl;
    }
    int value() const { return value_; }
 private:
    int value_{};
};

class TypeRegistry {
 protected:
  TypeRegistry(py::module m)
      : m_(m) {
    // Import modules into `locals_`.
    exec(R"""(
from __future__ import print_function
import numpy as np; import ctypes

def _get_type_name(t):
  prefix = t.__module__ + "."
  if prefix == "__main__.":
    prefix = ""
  return prefix + t.__name__

def print_(x): print(x)
)""");

    RegisterCommon();
  }

  static std::unique_ptr<TypeRegistry> instance_;
 public:

  static void Init(py::module m) {
    assert(instance_ == nullptr);
    instance_.reset(new TypeRegistry(m));
  }

  static const TypeRegistry& Instance() {
    assert(instance_);
    return *instance_;
  }

  template <typename T>
  py::handle get_py() {
    // Check if it's a custom-registered type.
    const std::type_info& tinfo = typeid(T);
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

  py::handle get_py_canonical(py::handle py_type) {
    // Since there's no easy / good way to expose C++ type id's to Python,
    // just canonicalize Python types.
    return py_to_py_canonical_.attr("get")(py_type, py_type);
  }

  py::str get_cpp_name(py::handle py_type) {
    py::handle py_type_fin = get_py_canonical(py_type);
    py::object out = py_name_.attr("get")(py_type_fin);
    if (out.is(py::none())) {
      out = eval("_get_type_name")(py_type_fin);
    }
    return out;
  }

 private:
  template <typename T>
  void Register(const std::string& py_values, const std::string& cpp_name = "") {
    size_t cpp_key = std::type_index(typeid(T)).hash_code();
    py::tuple py_types = eval(py_values);
    py::handle py_canonical = py_types[0];
    cpp_to_py_[cpp_key] = py_canonical;
    for (auto t : py_types) {
      py_to_py_canonical_[t] = py_canonical;
    }
    py_name_[py_canonical] = cpp_name.empty() ? nice_type_name<T>() : cpp_name;
  }

  py::object eval(const std::string& expr) {
    return py::eval(expr, py::globals(), locals_);
  }

  void exec(const std::string& expr) {
    py::exec(expr, py::globals(), locals_);
  }

  void RegisterCommon() {
    // Make mappings for C++ RTTI to Python types.
    // Unfortunately, this is hard to obtain from `pybind11`.
    Register<bool>("bool,");
    Register<std::string>("str,", "string");
    // Use numpy values by default.
    Register<double>("np.double, float, ctypes.c_double");
    Register<float>("np.float32, ctypes.c_float");
    Register<int>("np.int32, int, ctypes.c_int32");
  }

  py::handle m_;
  py::object locals_;

  std::map<size_t, py::handle> cpp_to_py_;
  py::dict py_to_py_canonical_;
  py::dict py_name_;
};

std::unique_ptr<TypeRegistry> TypeRegistry::instance_;


int main() {
  {
    py::scoped_interpreter guard{};

    // For now, just run with NumPyTypes.
    py::exec("import numpy as np");

    py::module m("check");

    TypeRegistry::Init(m);

    py::class_<TypeRegistry>(m, "TypeRegistry")
        .def_static("Instance", &TypeRegistry::Instance, py::return_value_policy::reference)
        .def("get_py_canonical", &TypeRegistry::get_py_canonical)
        .def("get_cpp_name", &TypeRegistry::get_cpp_name);

    py::class_<SimpleType>(m, "SimpleType")
        .def(py::init<int>())
        .def("value", &SimpleType::value);

    py::globals()["m"] = m;

    py::exec(R"""(
tr = m.TypeRegistry.Instance()
print(tr.get_py_canonical(int))
print(tr.get_cpp_name(float))
print(tr.get_py_canonical(m.SimpleType))
print(tr.get_cpp_name(m.SimpleType))
)""");
  }

  cout << "[ Done ]" << endl;

  return 0;
}
