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

class PyBase : public A {
 public:
  int pure(int value) override {
    PYBIND11_OVERLOAD(int, Base, pure, value);
  }
  int optional(int value) override {
    PYBIND11_OVERLOAD(int, Base, optional, value);
  }
};

class CppExtend : public A {
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

int call_method(A& base) {
  return base.dispatch(9);
}

PYBIND11_MODULE(_inherit_check, m) {
  m.doc() = "Simple check on inheritance";

  py::class_<A, PyBase> base(m, "Base");
  base
    .def(py::init<>())
    .def("pure", &Base::pure)
    .def("optional", &Base::optional)
    .def("dispatch", &Base::dispatch);

  py::class_<CppExtend>(m, "CppExtend", base)
    .def(py::init<>());

  m.def("call_method", &call_method);
}

}  // namespace inherit_check
