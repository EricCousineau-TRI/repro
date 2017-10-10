// Purpose: Test what avenues might be possible for creating instances in Python
// to then be owned in C++.

#include <cstddef>
#include <cmath>
#include <sstream>
#include <string>

#include <pybind11/cast.h>
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace py::literals;
using namespace std;

namespace ownership {

class A {
 public:
  A(int value)
      : value_(value) {
    cout << "A::A(" << value << ")" << endl;
  }
  virtual ~A() {
    cout << "A::~A()" << endl;
  }

  virtual int value() const { return value_; }

 private:
  int value_{};
};

unique_ptr<A> create_instance() {
  return make_unique<A>(50);
}

shared_ptr<A> check_creation(py::function py_factory, bool do_copy) {
  shared_ptr<A> in = py::cast<shared_ptr<A>>(py_factory());
  return in;
}

class PyA : public A {
 public:
  using A::A;
  ~PyA() {
    cout << "PyA::~PyA()" << endl;
  }
  int value() const override {
    PYBIND11_OVERLOAD(int, A, value);
  }
};

PYBIND11_MODULE(_ownership, m) {
  m.doc() = "Check ownership possibilites";

  py::class_<A, PyA, std::shared_ptr<A>>(m, "A")
    .def(py::init<int>())
    .def("value", &A::value);

  m.def("create_instance", &create_instance);
  m.def("check_creation", &check_creation);
}

}  // namespace scalar_type

int main(int, char**) {
  py::scoped_interpreter guard{};

  cout << "Start" << endl;

  py::module m("ownership");
  ownership::pybind11_init__ownership(m);

  cout << "Registered" << endl;

  py::dict globals = py::globals();
  globals["m"] = m;

  cout << "Eval" << endl;

  py::exec(R"""(
class Child(m.A):
  def __init__(self, value):
    m.A.__init__(self, value)
    print("Child.__init__({})".format(value))
  def __del__(self):
    print("Child.__del__")
  def value(self):
    print("Child.value")
    return 10 * m.A.value(self)
)""");

  py::exec(R"""(
factory = lambda: m.create_instance()
obj = m.check_creation(factory, False)
print(obj.value())
del obj

factory = lambda: Child(10)
obj = m.check_creation(factory, False)
print(obj.value())
del obj
)""", py::globals());

  cout << "Done" << endl;

  return 0;
}


/*
$ ./ownership_embed
Start
Registered
Eval
A::A(50)
50
A::~A()
A::A(10)
Child.__init__(10)
Child.__del__
10
PyA::~PyA()
A::~A()
Done
*/
