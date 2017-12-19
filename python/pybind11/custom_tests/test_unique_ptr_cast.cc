// Purpose: Test what avenues might be possible for creating instances in Python
// to then be owned in C++.

#include <cstddef>
#include <cmath>
#include <sstream>
#include <string>

#define PYBIND11_WARN_DANGLING_UNIQUE_PYREF

#include <pybind11/cast.h>
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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

class Base {
 public:
  Base(int value)
      : value_(value) {
    cout << "Base::Base(int)\n";
  }
  virtual ~Base() {
    cout << "Base::~Base()\n";
  }
  virtual int value() const {
    cout << "Base::value()\n";
    return value_;
  }
 private:
  int value_{};
};

class Child : public Base {
 public:
  Child(int value)
     : Base(value) {}
  ~Child() {
    cout << "Child::~Child()\n";
  }
  int value() const override {
    cout << "Child::value()\n";
    return 10 * Base::value();
  }
};

class ChildB : public Base {
 public:
  ChildB(int value)
     : Base(value) {}
  ~ChildB() {
    cout << "ChildB::~ChildB()\n";
  }
  int value() const override {
    cout << "ChildB::value()\n";
    return 10 * Base::value();
  }
};

// TODO(eric.cousineau): Add converter for `is_base<T, wrapper<T>>`, only for
// `cast` (C++ to Python) to handle swapping lifetime control.

// Trampoline class.
class PyBase : public py::wrapper<Base> {
 public:
  typedef py::wrapper<Base> TBase;
  using TBase::TBase;
  ~PyBase() {
    cout << "PyBase::~PyBase()" << endl;
  }
  int value() const override {
    PYBIND11_OVERLOAD(int, Base, value);
  }
};
class PyChild : public py::wrapper<Child> {
 public:
  typedef py::wrapper<Child> Base;
  using Base::Base;
  ~PyChild() {
    cout << "PyChild::~PyChild()" << endl;
  }
  int value() const override {
    PYBIND11_OVERLOAD(int, Child, value);
  }
};
class PyChildB : public py::wrapper<ChildB> {
 public:
  typedef py::wrapper<ChildB> Base;
  using Base::Base;
  ~PyChildB() {
    cout << "PyChildB::~PyChildB()" << endl;
  }
  int value() const override {
    PYBIND11_OVERLOAD(int, ChildB, value);
  }
};

unique_ptr<Base> check_creation(py::function create_obj) {
  // Test getting a pointer.
//  Base* in_test = py::cast<Base*>(obj);
  // Base a terminal pointer.
  // NOTE: This yields a different destructor order.
  // However, the wrapper class destructors should NOT interfere with nominal
  // Python destruction.
  cout << "---\n";
  unique_ptr<Base> fin = py::cast<unique_ptr<Base>>(create_obj());
  fin.reset();
  cout << "---\n";
  // Test pass-through.
  py::object obj = create_obj();
  unique_ptr<Base> in = py::cast<unique_ptr<Base>>(std::move(obj));
  return in;
}

unique_ptr<SimpleType> check_creation_simple(py::function create_obj) {
  cout << "---\n";
  unique_ptr<SimpleType> fin = py::cast<unique_ptr<SimpleType>>(create_obj());
  fin.reset();
  cout << "---\n";
  py::object obj = create_obj();
  unique_ptr<SimpleType> in = py::cast<unique_ptr<SimpleType>>(std::move(obj));
  return in;
}

// TODO(eric.cousineau): If a user uses `object` as a pass in, it should keep the reference count low
// (so that we can steal it, if need be).
// Presently, `pybind11` increases that reference count if `object` is an argument.

// Check casting.
unique_ptr<Base> check_cast_pass_thru(unique_ptr<Base> in) { //py::handle h) { //
//  py::object py_in = py::reinterpret_steal<py::object>(h);
//  auto in = py::cast<unique_ptr<Base>>(std::move(py_in));
  cout << "Pass through: " << in->value()<< endl;
  return in;
}

unique_ptr<Base> check_clone(unique_ptr<Base> in) {
//  auto in = py::cast<unique_ptr<Base>>(std::move(py_in));
  cout << "Clone: " << in->value()<< endl;
  unique_ptr<Base> out(new Base(20 * in->value()));
  return out;
}

unique_ptr<Base> check_new() {
    return make_unique<Base>(10);
}

class BaseContainer {
 public:
  Base* add(unique_ptr<Base> in) {
    Base* out = in.get();
    bases_.emplace_back(std::move(in));
    return out;
  }
  vector<Base*> list() const {
    vector<Base*> out;
    for (auto& ptr : bases_)
      out.push_back(ptr.get());
    return out;
  }
  vector<unique_ptr<Base>>& mutable_list() {
    return bases_;
  }
  vector<unique_ptr<Base>> release_list() {
    return std::move(bases_);
  }
 private:
  vector<unique_ptr<Base>> bases_;
};

void terminal_func(unique_ptr<Base> ptr) {
  cout << "Value: " << ptr->value() << endl;
  ptr.reset();  // This will destroy the instance.
  cout << "Destroyed in C++" << endl;
}

class Simple {};

void terminal_func_simple(unique_ptr<Simple> ptr) {
  ptr.reset();
}

unique_ptr<Simple> pass_thru_simple(unique_ptr<Simple> ptr) {
  return ptr;
}

PYBIND11_MODULE(_move, m) {
  py::class_<Simple>(m, "Simple")
      .def(py::init<>());
  m.def("terminal_func_simple", &terminal_func_simple);
  m.def("pass_thru_simple", &pass_thru_simple);

  py::class_<Base, PyBase>(m, "Base")
    .def(py::init<int>())
    .def("value", &Base::value);

  py::class_<BaseContainer>(m, "BaseContainer")
      .def(py::init<>())
      .def("add", &BaseContainer::add, py::return_value_policy::reference)
      .def("list", &BaseContainer::list)
//      .def("mutable_list", &BaseContainer::mutable_list)  // Not supported (as expected)
      .def("release_list", &BaseContainer::release_list);

  py::class_<Child, PyChild, Base>(m, "Child")
      .def(py::init<int>())
      .def("value", &Child::value);

  // NOTE: Not explicit calling `Base` as a base. Relying on Python downcasting via `py_type`.
  py::class_<ChildB, PyChildB>(m, "ChildB")
      .def(py::init<int>())
      .def("value", &ChildB::value);

  m.def("check_creation", &check_creation);
  m.def("check_cast_pass_thru", &check_cast_pass_thru);
  m.def("check_clone", &check_clone);
  m.def("check_new", &check_new);
  m.def("terminal_func", &terminal_func);

  // Make sure this setup doesn't botch the usage of `shared_ptr`, compile or run-time.
  class SharedClass {};
  py::class_<SharedClass, shared_ptr<SharedClass>>(m, "SharedClass");

  // Make sure this also still works with non-virtual, non-wrapper types.
  py::class_<SimpleType>(m, "SimpleType")
      .def(py::init<int>())
      .def("value", &SimpleType::value);
  m.def("check_creation_simple", &check_creation_simple);

  auto mdict = m.attr("__dict__");
  py::exec(R"(
class PyExtBase(Base):
    def __init__(self, value):
        Base.__init__(self, value)
        print("PyExtBase.PyExtBase")
    def __del__(self):
        print("PyExtBase.__del__")
    def value(self):
        print("PyExtBase.value")
        return 10 * Base.value(self)

class PyExtChild(Child):
    def __init__(self, value):
        Child.__init__(self, value)
        print("PyExtChild.PyExtChild")
    def __del__(self):
        print("PyExtChild.__del__")
    def value(self):
        print("PyExtChild.value")
        return Child.value(self)

class PyExtChildB(ChildB):
    def __init__(self, value):
        ChildB.__init__(self, value)
        print("PyExtChildB.PyExtChildB")
    def __del__(self):
        print("PyExtChildB.__del__")
    def value(self):
        print("PyExtChildB.value")
        return ChildB.value(self)
)", mdict, mdict);

    // Define move container thing
    py::exec(R"(
class PyMove:
    """ Provide a wrapper to permit passing an object to be owned by C++ """
    _is_move_container = True

    def __init__(self, obj):
        assert obj is not None
        self._obj = obj

    def release(self):
        from sys import getrefcount
        obj = self._obj
        self._obj = None
        ref_count = getrefcount(obj)
        # Cannot use `assert ...`, because it will leave a latent reference?
        # Consider a `with` reference?
        if ref_count > 2:
            obj = None
            raise AssertionError("Object reference is not unique, got {} extra references".format(ref_count - 2))
        else:
            assert ref_count == 2
            return obj
)", py::globals(), mdict);
}

// Export this to get access as we desire.
void custom_init_move(py::module& m) {
  PYBIND11_CONCAT(pybind11_init_, _move)(m);
}

void check_cpp_simple() {
  cout << "\n[ check_cpp_simple ]\n";
  py::exec(R"(
m.terminal_func_simple(m.Simple())
m.pass_thru_simple(m.Simple())
)");
}

void check_pure_cpp_simple() {
  cout << "\n[ check_pure_cpp_simple ]\n";
  py::exec(R"(
def create_obj():
    return [m.SimpleType(256)]
obj = m.check_creation_simple(create_obj)
print(obj.value())
del obj  # Calling `del` since scoping isn't as tight here???
)");
}

void check_pure_cpp() {
  cout << "\n[ check_pure_cpp ]\n";
  py::exec(R"(
def create_obj():
    return [m.Base(10)]
obj = m.check_creation(create_obj)
print(obj.value())
del obj
)");
}

void check_pass_thru() {
    cout << "\n[ check_pure_cpp ]\n";

    py::exec(R"(
obj = m.check_cast_pass_thru([m.Base(20)])
print(obj.value())
del obj

obj = m.check_clone([m.Base(30)])
print(obj.value())
del obj
)");
}

void check_py_child() {
  // Check ownership for a Python-extended C++ class.
  cout << "\n[ check_py_child ]\n";
  py::exec(R"(
def create_obj():
    return [m.PyExtBase(20)]
obj = m.check_creation(create_obj)
print(obj.value())
del obj
)");
}

void check_casting() {
  // Check a class which, in C++, derives from the direct type, but not the alias.
  cout << "\n[ check_casting ]\n";
  py::exec(R"(
def create_obj():
    return [m.PyExtChild(30)]
obj = m.check_creation(create_obj)
print(obj.value())
del obj
)");
}

void check_casting_without_explicit_base() {
  // Check a class which, in C++, derives from the direct type, but not the alias.
  cout << "\n[ check_casting_without_explicit_base ]\n";
  py::exec(R"(
def create_obj():
    return [m.PyExtChildB(30)]
obj = m.check_creation(create_obj)
print(obj.value())
del obj
)");
}

void check_container() {
  cout << "\n[ check_container ]\n";
  py::exec(R"(
print("Create container")
c = m.BaseContainer()
print("Create instance")
b1 = m.PyExtChildB(30)
print("Add instance")
b1 = c.add(b1)
print("Create instance 2")
b2 = m.Base(10)
print("Add instance 2")
b2 = c.add(b2)
print("Print values")
print(b1.value())
print(b2.value())

print("Delete references")
del b1
del b2

print("Delete container")
del c

print("Make new container")
c = m.BaseContainer()
b1 = c.add(m.PyExtChildB(30))
b2 = c.add(m.Base(10))
print("Delete references - have to be strict...")
del b1
del b2
print("Release to list")
li = c.release_list()
del c
print("List: {}".format(li))
for x in li:
    print("value: {}".format(x.value()))
    # del x
print("Delete list")
del li
)");
}

void check_terminal() {
    cout << "\n[ check_terminal ]\n";
    py::exec(R"(
m.terminal_func([m.PyExtBase(20)])
)");
}

int main() {
  {
    py::scoped_interpreter guard{};

    py::module m("_move");
    custom_init_move(m);
    py::globals()["m"] = m;

    check_cpp_simple();
    check_pass_thru();
    check_pure_cpp_simple();
    check_pure_cpp();
    check_py_child();
    check_casting();
    check_casting_without_explicit_base();
    check_terminal();
    check_container();
  }

  cout << "[ Done ]" << endl;

  return 0;
}
