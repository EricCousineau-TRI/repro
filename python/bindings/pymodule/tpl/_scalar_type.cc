// Purpose: Test what binding different scalar types (template arguments) might
// look like with `pybind11`.
// Specifically, having a base class of <T, U>, and seeing if pybind11 can bind
// it "easily".

#include <cstddef>
#include <cmath>
#include <sstream>
#include <string>

#include <pybind11/cast.h>
#include <pybind11/eval.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpp/name_trait.h"

namespace py = pybind11;
using namespace py::literals;
using namespace std;

namespace scalar_type {

template <typename T = float, typename U = int16_t>
class Base;

}

using scalar_type::Base;
NAME_TRAIT_TPL(Base)

namespace scalar_type {

// Simple base class.
template <typename T, typename U>
class Base {
 public:
  Base(T t, U u)
    : t_(t),
      u_(u) {}
  template <typename Tc, typename Uc>
  Base(const Base<Tc, Uc>& other)
    : Base(static_cast<T>(other.t_),
           static_cast<U>(other.u_)) {}

  virtual U pure(T value) = 0;
  virtual U optional(T value) {
    cout << py_name() << endl;
    return static_cast<U>(value);
  }

  int dispatch(int value) {
    cout << "cpp.dispatch:\n";
    cout << "  ";
    int pv = pure(value);
    cout << "  ";
    int ov = optional(value);
    return pv + ov;
  }

  // TODO: Use `typeid()` and dynamic dispatching?
  static string py_name() {
    return "Base__T_" + name_trait<T>::name() +
      "__U_" + name_trait<U>::name();
  }

 private:
  template <typename Tc, typename Uc> friend class Base;
  T t_{};
  U u_{};
};

template <typename T, typename U>
class PyBase : public Base<T, U> {
 public:
  typedef Base<T, U> B;

  using B::B;

  U pure(T value) override {
    PYBIND11_OVERLOAD_PURE(U, B, pure, value);
  }
  U optional(T value) override {
    PYBIND11_OVERLOAD(U, B, optional, value);
  }
};

template <typename T, typename U>
void call_method(const Base<T, U>& base) {
  base.dispatch();
}

template <typename T, typename U>
void register_base(py::module m) {
  string name = Base<T, U>::py_name();
  typedef Base<T, U> C;
  typedef PyBase<T, U> PyC;
  py::class_<C, PyC> base(m, name.c_str());
  base
    .def(py::init<T, U>())
    .def("pure", &C::pure)
    .def("optional", &C::optional)
    .def("dispatch", &C::dispatch);

  // // Can't figure this out...
  // m.def("call_method", py::overload_cast<const Base<T, U>&>(&call_method));

  // http://pybind11.readthedocs.io/en/stable/advanced/pycpp/object.html#casting-back-and-forth
  // http://pybind11.readthedocs.io/en/stable/advanced/pycpp/utilities.html
  // http://pybind11.readthedocs.io/en/stable/advanced/misc.html
  auto locals = py::dict("cls"_a=base);
  auto globals = m.attr("__dict__");
  py::eval<py::eval_statements>(R"(#
cls.stuff = 10
)", globals, locals);

  // // Register the type in Python.
  // // TODO: How to execute Python with arguments?
  // // How to convert a C++ typeid to a Python type, and vice versa?
  // // py::detail::type_info, detail::get_type_info
  // auto cls = py::exec();
  // auto locals = py::dict("params"_a=types, "cls"_a=cls);
  // py::exec(R"(
  //   _base_types[
  // )", m);
}

PYBIND11_PLUGIN(_scalar_type) {
  py::module m("_scalar_type", "Simple check on scalar / template types");

  auto globals = m.attr("__dict__");
  py::eval<py::eval_statements>(R"(#
# Dictionary.
#   Key: (T, U)
#   Value: PyType
base_types = {}
)", globals);

  register_base<float, int16_t>(m);
  register_base<double, int64_t>(m);

  return m.ptr();
}

}  // namespace scalar_type
