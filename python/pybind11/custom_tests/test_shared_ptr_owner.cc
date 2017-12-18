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

int global = 0;

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

  void debug_hook() {
    global += 1;
  }

 private:
  int value_{};
};

class AContainer {
 public:
    shared_ptr<A> add(shared_ptr<A> a) {
      as_.push_back(a);
      return as_.back();
    }
    vector<shared_ptr<A>> release_list() {
      return std::move(as_);
    }
 private:
    vector<shared_ptr<A>> as_;
};

unique_ptr<A> create_instance() {
  return make_unique<A>(50);
}

shared_ptr<A> check_creation(py::function py_factory, bool do_copy) {
  py::object obj = py_factory();
  cout << "-- C++ value --" << endl;
  cout << obj.attr("value")().cast<int>() << endl;
  shared_ptr<A> in = py::cast<shared_ptr<A>>(obj);
  return in;
}

class PyA : public py::wrapper<A> {
 public:
  using py::wrapper<A>::wrapper;
  ~PyA() {
    cout << "PyA::~PyA()" << endl;
  }
  int value() const override {
    cout << "PyA::value()" << endl;
    PYBIND11_OVERLOAD(int, A, value);
  }
};

PYBIND11_MODULE(_ownership, m) {
  m.doc() = "Check ownership possibilites";

  py::class_<A, PyA, std::shared_ptr<A>>(m, "A")
    .def(py::init<int>())
    .def("value", &A::value)
    .def("debug_hook", &A::debug_hook);

  py::class_<AContainer, shared_ptr<AContainer>>(m, "AContainer")
    .def(py::init<>())
    .def("add", &AContainer::add)
    .def("release_list", &AContainer::release_list);

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
class ChildParent(m.A):
  def __init__(self, value):
    m.A.__init__(self, value)
    self.extra = [value * 2]
    print("Child.__init__({})".format(value))
  def __del__(self):
    m.A.debug_hook(self)
    print("Child.__del__")
  def value(self):
    # TODO(eric.cousineau): Fix this...
    print("Child.value() (extra = {})".format(self.extra))
    return 10 * m.A.value(self)

class Child(ChildParent):
  def __init__(self, value):
    ChildParent.__init__(self, value)
    self.sub_extra = [value * 3]
  def __del__(self):
    print("Sub-Child.__del__")
    ChildParent.__del__(self)
    self.debug_hook()
  def value(self):
    print("Sub-Child.value() (sub_extra = {})".format(self.sub_extra))
    return 10 * ChildParent.value(self)

"""
print("Attempt to reassign class-level __del__")
del_orig = Child.__del__
def del_new(self):
    print("wrapped!")
    print(del_orig)
    del_orig(self)
Child.__del__ = del_new  # This works if we override `py_type->tp_del`?
print("Done")

print("One more time")
obj = Child(10)
obj.__del__ = lambda: del_new(obj)
print("Done again")
del obj
"""
)""");

//  py::exec(R"""(
//factory = lambda: m.create_instance()
//obj = m.check_creation(factory, False)
//print(obj.value())
//del obj
//)""");
//
//  py::exec(R"""(
//print("---")
//c = Child(30)
//factory = lambda: c
//obj = m.check_creation(factory, False)
//print("-- Python value --")
//print(obj.value())
//del obj
//del factory
//del c
//)""");
//
//  py::exec(R"""(
//print("---")
//factory = lambda: Child(10)
//obj = m.check_creation(factory, False)
//print("-- Python value --")
//print(obj.value())
//del obj
//)""");

  py::exec(R"""(
print("---")
c = m.AContainer()
# Pass back through to reclaim
out = c.add(Child(20))
# See what happens if the value is destructed
print("-- Python value 1 --")
print(out.value())
del out
print("Get list")
li = c.release_list()
print("-- Python value 2 --")
print(li[0].value())
print("Remove container(s)")
del c
del li
)""");

  cout << "Done" << endl;

  return 0;
}

/*
Start
Registered
Eval
A::A(50)
-- C++ value --
50
50
A::~A()
---
A::A(30)
Child.__init__(30)
-- C++ value --
Child.value()
PyA::value()
300
-- Python value --
Child.value()
PyA::value()
300
Child.__del__
PyA::~PyA()
A::~A()
---
A::A(10)
Child.__init__(10)
-- C++ value --
Child.value()
PyA::value()
100
-- Python value --
Child.value()
PyA::value()
100
Child.__del__
PyA::~PyA()
A::~A()
---
A::A(20)
Child.__init__(20)
-- Python value 1 --
Child.value()
PyA::value()
200
Child.__del__
SharedPtr holder has use_count() > 1 on destruction for a Python-derived class.
Attempting to interrupt
Interrupting destruction
-- Python value 2 --
Child.value()
PyA::value()
200
Remove container(s)
Child.__del__
PyA::~PyA()
A::~A()
Done
*/
