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
#include "python/bindings/pymodule/tpl/cpp_tpl.h"
#include "python/bindings/pymodule/tpl/simple_converter_py.h"

namespace py = pybind11;
using namespace py::literals;
using namespace std;

using namespace simple_converter;

namespace scalar_type {

template <typename T = float, typename U = int16_t>
class Base;

}

using scalar_type::Base;
NAME_TRAIT_TPL(Base)

namespace scalar_type {

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
    return TypeRegistry::GetPyInstance().GetName<Base>();
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

template <typename T, typename U>
void print_base_name() {
  std::cout << "print_base_name: " << Base<T, U>::py_name() << std::endl;
}

template <bool Value>
void template_bool() {
  std::cout << "template_bool: " << Value << std::endl;
}

template <int N>
void template_int() {
  std::cout << "template_int: " << N << std::endl;
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

template <typename ... Ts>
using single_type_pack_list =
    type_pack<type_pack<Ts>...>;

template <typename T, T ... Values>
using single_type_pack_sequence =
    single_type_pack_list<std::integral_constant<T, Values>...>;


PYBIND11_MODULE(_scalar_type, m) {
  m.doc() = "Simple check on scalar / template types";

  // Add instantiations and conversion mechanisms.
  using ParamList = type_pack<
      type_pack<int, double>,
      type_pack<double, int>>;

  py::object tpl = InitOrGetTemplate(m, "BaseTpl", "TemplateClass");
  RegisterConverter<BaseConverter>(m, tpl);

  {
    auto inst = [&m, tpl](auto param) {
      // Extract parameters.
      using Param = decltype(param);
      using T = typename Param::template type<0>;
      using U = typename Param::template type<1>;
      // Typedef classes.
      using BaseT = Base<T, U>;
      using PyBaseT = PyBase<T, U>;
      // Define class.
      string name = nice_type_name<BaseT>();
      // N.B. This  name will be overwritten by `tpl.add_class(...)`.
      py::class_<BaseT, PyBaseT> py_class(m, name.c_str());
      py_class
        .def(py::init<T, U, std::unique_ptr<BaseConverter>>(),
             py::arg("t"), py::arg("u"), py::arg("converter") = nullptr)
        .def("t", &BaseT::t)
        .def("u", &BaseT::u)
        .def("pure", &BaseT::pure)
        .def("optional", &BaseT::optional)
        .def("dispatch", &BaseT::dispatch);
      AddInstantiation(tpl, py_class, param);

      // Can't get `overload_cast` to infer `Return` type.
      // Have to explicitly cast... :(
      m.def("call_method", static_cast<void(*)(const BaseT&)>(&call_method));

      // Add template methods for `DoTo`.
      {
        auto inst = [&](auto to_param) {
          using To = typename decltype(to_param)::template bind<Base>;
          // WARNING: If you forget `param`, then it'll assume an empty set...
          // :(
          AddTemplateMethod(
              py_class, "DoTo", &BaseT::template DoTo<To>, to_param);
        };
        IterTemplate(inst, ParamList{});
      }

      // Register conversions.
      RegisterConversions<Base, BaseConverter>(
          py_class, tpl, param, ParamList{});
    };
    IterTemplate(inst, ParamList{});
    // Default instantiation.
    m.attr("Base") = tpl.attr("get_instantiation")();
  }

  m.def("do_convert", &do_convert);
  m.def("take_ownership", &take_ownership);

  // Literals.
  // - Manual.
  AddTemplateFunction<int, double>(
      m, "print_base_name", &print_base_name<int, double>);
  AddTemplateFunction<double, int>(
      m, "print_base_name", &print_base_name<double, int>);
  {
    // - Looping.
    auto inst = [&m](auto param) {
      using Param = decltype(param);
      constexpr auto Value = Param::template type<0>::value;
      AddTemplateFunction(
          m, "template_bool", &template_bool<Value>, param);
    };
    using ParamList = single_type_pack_sequence<bool, false, true>;
    IterTemplate(inst, ParamList{});
  }
  {
    auto inst = [&m](auto param) {
      using Param = decltype(param);
      constexpr auto N = Param::template type<0>::value;
      AddTemplateFunction(
          m, "template_int", &template_int<N>, param);
    };
    using ParamList = single_type_pack_sequence<int, 0, 1, 2, 5>;
    IterTemplate(inst, ParamList{});
  }
}

}  // namespace scalar_type
