// Purpose: Base what avenues might be possible for creating instances in Python
// to then be owned in C++.

#include <cstddef>
#include <cmath>
#include <sstream>
#include <string>

#include <pybind11/cast.h>
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/pybind11.h>

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
 public:
  void TypeRegistry(py::module m)
      : m_(m) {
    // Import modules.
    eval("import numpy as np; import ctypes");
  }

  template <typename T>
  py::handle get_py() {
    // Check if it's a custom-registered type.
    std::type_info& tinfo = tyipeid(T);
    size_t cpp_key = std::type_index(tinfo).hash_code();
    auto iter = cpp_to_py_.find(cpp_key);
    if (iter != cpp_to_py_.end()) {
      return iter->second;
    } else {
      // Get from pybind11-registered types.
      // WARNING: Internal API :(
      auto* info = py::detail::get_type_info(tinfo);
      assert(info != nullptr);
      return py::handle(info->type);
    }
  }

  py::handle get_py_canonical(py::handle py_type) {
    // Since there's no easy / good way to expose C++ type id's to Python,
    // just canonicalize Python types.
    size_t cpp_key 
  }

 private:
  template <typename T>
  void Register(const std::string& py_values) {
    size_t cpp_key = std::type_index(typeid(T)).hash_code();
    py::tuple py_types = eval(py_values);
    py::handle py_canonical = py_types[0];
    cpp_to_py_[cpp_key] = py_canonical;
    for (auto t : py_types) {
      py_to_py_canonical[t] = py_canonical;
    }
    std::string cpp_name = nice_name(T);
    py_name_[py_canonical] = cpp_name;
  }

  auto eval(const std::string& expr) {
    return py::eval(expr, py::globals(), locals_);
  }

  void RegisterCommon() {
    // Make mappings for C++ RTTI to Python types.
    // Unfortunately, this is hard to obtain from `pybind11`.
    Register<bool>("bool,");
    Register<std::string>("str,");
    // Use numpy values by default.
    Register<double>("np.double, float, ctypes.c_double");
    Register<float>("np.float, ctypes.c_double");
    Register<int>("np.int32, int, ctypes.c_int32");
  }

  py::handle m_;
  py::object locals_;

  std::map<size_t, py::handle> cpp_to_py_;
  py::dict py_to_py_canonical_;
  py::dict py_name_;
};

void do_register() {
}


int main() {
  {
    py::scoped_interpreter guard{};

    // For now, just run with NumPyTypes.
    auto np = py::module::import("numpy");

    py::module m("check");
    py::class_<SimpleType>(m, "SimpleType")
        .def(py::init<int>())
        .def("value", &SimpleType::value);
    m.def("check", &check);


    py::globals()["m"] = m;

  }

  cout << "[ Done ]" << endl;

  return 0;
}
