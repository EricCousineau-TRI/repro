#include <iostream>
#include <string>

#include <pybind11/pybind11.h>

#include "cpp/name_trait.h"
#include "python/bindings/pymodule/const/cpp_const_pybind.h"

struct Test {
 public:
  Test(int value) : value_(value) {}

  int value() const { return value_; }
  void set_value(int value) { value_ = value; }

  void check_mutate() {}
  void check_const() const {}

  const Test* as_const() const { return this; }

 private:
  int value_{};
};

void func_mutate(Test* obj) {
  obj->check_mutate();
}

void func_const(const Test* obj) {
  obj->check_const();
}

PYBIND11_MODULE(_cpp_const_pybind_py, m) {
  m.doc() = "C++ Const Test";

  // m.def("func_mutate", WrapRef(&func_mutate));
  // m.def("func_const", WrapRef(&func_const));

  // py::class_<Test> test(m, "Test");
  // test
  //     .def(py::init<int>())
  //     .def("check_mutate", WrapRef(&Test::check_mutate))
  //     .def("check_const", WrapRef(&Test::check_const))
  //     .def("as_const", WrapRef(&Test::as_const))  // Should not need any rvp.
  //     .def_property(
  //         "value", WrapRef(&Test::value), WrapRef(&Test::set_value));
}
