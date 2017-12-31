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
#include "python/bindings/pymodule/tpl/cpp_tpl_types.h"

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

  // TODO: Use `typeid()` and dynamic dispatching?
  static string py_name() {
    return "BaseTpl[" + name_trait<T>::name() +
      ", " + name_trait<U>::name() + "]";
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

template <typename ... Args>
inline py::tuple get_py_types(type_pack<Args...> = {}) {
  return TypeRegistry::GetPyInstance().GetPyTypes<Args...>();
}

template <typename MetaPack, typename InstantiationFunc>
void RegisterInstantiations(
    py::object tpl, const InstantiationFunc& instantiation_func,
    MetaPack packs = {}) {
  MetaPack::template visit_lambda<no_wrap>(
      [&](auto pack) {
        // Register instantiation in `pybind`, using lambda
        // `auto`-friendly syntax., indexing by canonical Python types.
        tpl.attr("add_instantiation")(
            get_py_types(pack),
            instantiation_func(pack));
      });
}

template <typename Converter>
auto RegisterConverter(py::module m, py::object tpl) {
  // Register converter. Name does not matter.
  py::class_<Converter> converter(m, "ConverterTmp");
  converter
    .def(py::init<>())
    .def(
      "Add",
      [tpl](Converter* self,
             py::tuple params_to, py::tuple params_from,
             py::function py_converter) {
        // @pre `params_to` and `params_from` are canonical Python types.
        // Find conversion function using these types.
        py::tuple key = py::make_tuple(params_to, params_from);
        py::object add_py_converter = tpl.attr("_add_py_converter_map")[key];
        // Add the casted converter.
        add_py_converter(self, py_converter);
      });
  tpl.attr("Converter") = converter;
  tpl.attr("_add_py_converter_map") = py::dict();
}

template <
    template <typename...> class Tpl, typename Converter,
    template <typename...> class Ptr = std::unique_ptr,
    typename ToPack = void, typename FromMetaPack = void,
    typename Check = is_different_from<ToPack>,
    // Use `void` here since these will be inferred, but allow the check to
    // have a default value.
    typename PyClass = void>
void RegisterConversions(
    PyClass& py_class, py::object tpl,
    ToPack to_pack = {}, FromMetaPack from_packs = {}) {
  using To = typename ToPack::template bind<Tpl>;
  py::tuple to_tup = get_py_types(ToPack{});
  FromMetaPack::template visit_lambda_if<Check, no_wrap>(
      [&](auto from_pack) {
        // Register base conversion.
        using FromPack = decltype(from_pack);
        using From = typename FromPack::template bind<Tpl>;
        auto from_param = get_py_types(FromPack{});
        // Add Python converter function, but bind using Base C++ overloads via
        // pybind.
        auto add_py_converter = [](Converter* converter, py::function py_func) {
          // Wrap with C++ type information.
          using ConversionFunc = std::function<Ptr<To> (const From&)>;
          auto cpp_func = py::cast<ConversionFunc>(py_func);
          // Add using overload.
          converter->Add(cpp_func);
        };
        // Register function dispatch.
        py::tuple key = py::make_tuple(to_tup, from_param);
        tpl.attr("_add_py_converter_map")[key] =
            py::cpp_function(add_py_converter);
        // Add Python conversion.
        py_class.def(py::init<const From&>());
      });
}

template <
    typename PyClass, typename MetaPack,
    typename InstantiationFunc>
py::object RegisterTemplateMethod(
    PyClass& py_class, const std::string& name,
    const InstantiationFunc& instantiation_func, MetaPack packs = {}) {
  py::handle TemplateMethod =
      py::module::import("pymodule.tpl.py_tpl").attr("TemplateMethod");
  std::string tpl_attr = "_tpl_" + name;
  py::object tpl = py::getattr(py_class, tpl_attr.c_str(), py::none());
  using Class = typename PyClass::type;
  if (tpl.is(py::none())) {
    // Add template backend.
    tpl = TemplateMethod(name, py_class);
    py::setattr(py_class, tpl_attr.c_str(), tpl);
    // Add read-only property.
    py_class.def_property(
        name.c_str(),
        [tpl](Class* self) {
          return tpl.attr("bind")(self);
        },
        // TODO: Fix this once pybind is upgraded.
        [](Class* self, py::handle) {
          throw std::runtime_error("Read-only property");
        });
  }
  // Ensure that pybind is aware that it's a function.
  auto cpp_instantiation_func = [instantiation_func](auto pack) {
    return py::cpp_function(instantiation_func(pack));
  };
  RegisterInstantiations(tpl, cpp_instantiation_func, packs);
  return tpl;
}


// TODO: Figure out how to handle literals...


PYBIND11_MODULE(_scalar_type, m) {
  m.doc() = "Simple check on scalar / template types";

  py::handle TemplateClass =
      py::module::import("pymodule.tpl.py_tpl").attr("TemplateClass");

  py::object tpl = TemplateClass("BaseTpl");
  m.attr("BaseTpl") = tpl;
  RegisterConverter<BaseConverter>(m, tpl);

  // Add instantiations and conversion mechanisms.
  using AllPack = type_pack<
      type_pack<int, double>,
      type_pack<double, int>>;

  auto base_instantiation = [&m, tpl](auto param_pack) {
    // Extract parameters.
    using Pack = decltype(param_pack);
    using T = typename Pack::template type<0>;
    using U = typename Pack::template type<1>;
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

    // Can't get `overload_cast` to infer `Return` type.
    // Have to explicitly cast... :(
    m.def("call_method", static_cast<void(*)(const BaseT&)>(&call_method));

    // Add template methods for `DoTo`.
    auto do_to_instantiation = [](auto to_pack) {
      using To = typename decltype(to_pack)::template bind<Base>;
      return [](BaseT* self) { return self->template DoTo<To>(); };
    };
    RegisterTemplateMethod(py_class, "DoTo", do_to_instantiation, AllPack{});

    // Register conversions.
    RegisterConversions<Base, BaseConverter>(
        py_class, tpl, param_pack, AllPack{});
    return py_class;
  };

  RegisterInstantiations(tpl, base_instantiation, AllPack{});

  // Default instantiation.
  m.attr("Base") = tpl.attr("get_instantiation")();

  m.def("do_convert", &do_convert);
  m.def("take_ownership", &take_ownership);
}

}  // namespace scalar_type
