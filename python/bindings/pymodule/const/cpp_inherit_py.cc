#include <iostream>

#include <pybind11/pybind11.h>

using namespace std;

// template <typename ... Args, typename Return, typename Class>
// auto virtual_overload((

class Base {
 public:
  virtual int stuff() const { return 1; }
};

class Child : public Base {
 public:
  int stuff() const override { return 10; }
};

namespace py = pybind11;

PYBIND11_MODULE(_cpp_inherit, m) {
  py::class_<Base>(m, "Base")
    .def("stuff", &Base::stuff);

  py::class_<Child, Base>(m, "Child");

  m.def("create_base", []() { return new Base(); });
  m.def("create_child", []() -> Base* { return new Child(); });

  auto a = &Base::stuff;
  auto b = &Child::stuff;

  Child c;
  cout << (c.*a)() << endl;
  cout << (c.*b)() << endl;
}
