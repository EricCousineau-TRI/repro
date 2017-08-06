#include <cstddef>
#include <cmath>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace std;

namespace inherit_check {

// Simple base class.
class Base {
 public:
  // Base() {
  //   cout << "cpp.ctor" << endl;
  // }
  virtual int pure(int value) { return 0; }
  virtual int optional(int value) {
    return 0;
  }
  int dispatch(int value) {
    cout << "cpp.dispatch:\n";
    cout << "  ";
    int pv = pure(value);
    cout << "  ";
    int ov = optional(value);
    return pv + ov;
  }
};

class PyBase : public Base {
 public:
  int pure(int value) override {
    PYBIND11_OVERLOAD(int, Base, pure, value);
  }
  int optional(int value) override {
    PYBIND11_OVERLOAD(int, Base, optional, value);
  }
};

class CppExtend : public Base {
 public:
  int pure(int value) override {
    cout << "cpp.pure=" << value << endl;
    return value;
  }
  int optional(int value) override {
    cout << "cpp.optional=" << value << endl;
    return 10 * value;
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
