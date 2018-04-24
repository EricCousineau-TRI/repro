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
#include <pybind11/functional.h>

#include "cpp/name_trait.h"
#include "cpp/simple_converter.h"
#include "python/bindings/pymodule/tpl/cpp_template.h"
#include "python/bindings/pymodule/tpl/simple_converter_py.h"

namespace py = pybind11;
using namespace py::literals;
using namespace std;

using namespace simple_converter;

namespace scalar_type {

// Scalar type conversion.
template <typename T, typename U>
class Base;

typedef SimpleConverter<Base> BaseConverter;

// Simple base class.
template <typename T, typename U>
class Base {
 public:
  Base(T t, U u, std::unique_ptr<BaseConverter> converter = nullptr)
    : t_(t),
      u_(u),
      converter_(std::move(converter)) {
    if (!converter_) {
      converter_.reset(new BaseConverter());
      typedef Base<double, int> A;
      typedef Base<int, double> B;
      converter_->AddCopyConverter<A, B>();
      converter_->AddCopyConverter<B, A>();
    }
  }

  template <typename Tc, typename Uc>
  Base(const Base<Tc, Uc>& other)
    : Base(static_cast<T>(other.t_),
           static_cast<U>(other.u_),
           std::make_unique<BaseConverter>(*other.converter_)) {}

  virtual ~Base() {
    cout << "Base::~Base" << endl;
  }

  T t() const { return t_; }
  U u() const { return u_; }

  virtual U pure(T value) const { return U{}; } // = 0 -- Do not use for concrete converter example.
  virtual U optional(T value) const {
    cout << py_name() << endl;
    return static_cast<U>(value);
  }

  U dispatch(T value) const {
    cout << "cpp.dispatch [" << py_name() << "]:\n";
    cout << "value = " << value << endl;
    cout << " .t = " << t() << endl;
    cout << " .u = " << u() << endl;
    cout << "  ";
    U pv = pure(value);
    cout << "  = " << pv << endl;
    cout << "  ";
    U ov = optional(value);
    cout << "  = " << ov << endl;
    return pv + ov;
  }

  static string py_name() {
    return get_py_name<Base>();
  }

  template <typename To>
  std::unique_ptr<To> DoTo() const {
    return converter_->Convert<To>(*this);
  }

 private:
  template <typename Tc, typename Uc> friend class Base;

  T t_{};
  U u_{};
  std::unique_ptr<BaseConverter> converter_;
};


template <typename T, typename U>
class PyBase : public py::wrapper<Base<T, U>> {
 public:
  typedef Base<T, U> BaseT;
  typedef py::wrapper<Base<T, U>> BaseW;

  using BaseW::BaseW;

  U pure(T value) const override {
    // Do NOT use `BWrap` here as pybind uses direct RTTI on the supplied type.
    PYBIND11_OVERLOAD_PURE(U, BaseT, pure, value);
  }
  U optional(T value) const override {
    PYBIND11_OVERLOAD(U, BaseT, optional, value);
  }
};


template <typename T, typename U>
void call_method(const Base<T, U>& base) {
  base.dispatch(T{10});
}

std::unique_ptr<Base<double, int>> do_convert(const Base<int, double>& value) {
  cout << "Attempt conversion" << endl;
  std::unique_ptr<Base<double, int>> out(value.DoTo<Base<double, int>>());
  // auto out = std::make_unique<Base<double, int>>(8.5, 10);  // Not equivalent...
  // Try to create an instance of `ChildTpl`.
  cout << "Got it" << endl;
  return out;
}

// How can this work?
std::unique_ptr<Base<double, int>> take_ownership(py::function factory) {
  cout << "cpp call" << endl;
  py::object out_py = factory();
  cout << "cpp convert" << endl;
  return py::cast<std::unique_ptr<Base<double, int>>>(std::move(out_py));
}

PYBIND11_MODULE(_scalar_type_test, m) {
  m.doc() = "Simple check on scalar / template types";

  // Add instantiations and conversion mechanisms.
  {
    using ParamList = type_pack<
      type_pack<int, double>,
      type_pack<double, int>>;

    auto inst = [&m](auto param) {
      // Extract parameters.
      using Param = decltype(param);
      using T = typename Param::template type_at<0>;
      using U = typename Param::template type_at<1>;
      // Typedef classes.
      using BaseT = Base<T, U>;
      using PyBaseT = PyBase<T, U>;
      // Define class.
      py::class_<BaseT, PyBaseT> py_class(
          m, (std::string("_") + typeid(BaseT).name()).c_str());
      py_class
        .def(py::init<T, U, std::unique_ptr<BaseConverter>>(),
             py::arg("t"), py::arg("u"), py::arg("converter") = nullptr)
        .def("t", &BaseT::t)
        .def("u", &BaseT::u)
        .def("pure", &BaseT::pure)
        .def("optional", &BaseT::optional)
        .def("dispatch", &BaseT::dispatch);
      py::object tpl = AddTemplateClass(m, "BaseTpl", py_class, param, "Base");

      // Can't get `overload_cast` to infer `Return` type.
      // Have to explicitly cast... :(
      m.def("call_method", static_cast<void(*)(const BaseT&)>(&call_method));

      // Add template methods for `DoTo` and conversion.
      {
        auto inner = [&](auto inner_param) {
          using BaseInner = decltype(type_bind<Base>(inner_param));
          AddTemplateMethod(
              py_class, "DoTo", &BaseT::template DoTo<BaseInner>, inner_param);
          AddConversion<BaseConverter, BaseT, BaseInner>(py_class, tpl);
        };
        // Use `check_different_from` to avoid implicitly-deleted copy
        // constructor.
        type_visit(inner, ParamList{}, check_different_from<Param>{});
      }
    };
    type_visit(inst, ParamList{});
  }

  m.def("do_convert", &do_convert);
  m.def("take_ownership", &take_ownership);
}

}  // namespace scalar_type
