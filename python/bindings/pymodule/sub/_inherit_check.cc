#include <cstddef>
#include <cmath>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using std::string;
using std::ostringstream;

namespace inherit_check {

// Simple base class.
class Base {
 public:
  virtual string pure(const string& value) = 0;
  virtual string optional(const string& value) {
    return "";
  }
  string dispatch(const string& value) {
    return "cpp.dispatch: " + pure(value) + " " + optional(value);
  }
};

class PyBase : public Base {
 public:
  string pure(const string& value) override {
    PYBIND11_OVERLOAD_PURE(string, Base, pure, value);
  }
  string optional(const string& value) override {
    PYBIND11_OVERLOAD(string, Base, optional, value);
  }
};

class CppExtend : public Base {
 public:
  string pure(const string& value) override {
    return "cpp.pure=" + value;
  }
  string optional(const string& value) override {
    return "cpp.optional=" + value;
  }
};

PYBIND11_PLUGIN(_inherit_check) {
  py::module m("_inherit_check", "Simple check on inheritance");

  py::class_<Base, PyBase> base(m, "Base");
  base
    .def(py::init<>())
    .def("pure", &Base::pure)
    .def("optional", &Base::optional)
    .def("dispatch", &Base::dispatch);

  py::class_<CppExtend>(m, "CppExtend", base)
    .def(py::init<>());

  return m.ptr();
}

}  // namespace inherit_check
