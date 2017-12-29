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
    return "Base[" + name_trait<T>::name() +
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
    PYBIND11_OVERLOAD(U, BaseT, pure, value);
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

template <typename T, typename U>
void BaseTplInstantiation(py::module m, py::object tpl) {
  typedef Base<T, U> Cls;
  typedef PyBase<T, U> PyCls;
  // This name will be overwritten.
  string name = nice_type_name<Cls>();
  py::class_<Cls, PyCls> base(m, name.c_str());
  base
    .def(py::init<T, U, std::unique_ptr<BaseConverter>>(),
         py::arg("t"), py::arg("u"), py::arg("converter") = nullptr)
    .def("t", &Cls::t)
    .def("u", &Cls::u)
    .def("pure", &Cls::pure)
    .def("optional", &Cls::optional)
    .def("dispatch", &Cls::dispatch);

  const TypeRegistry& type_registry = TypeRegistry::GetPyInstance();

  // Register template class.
  auto type_tup = type_registry.GetPyTypes<T, U>();
  tpl.attr("add_class")(type_tup, base);

  // Can't get `overload_cast` to infer `Return` type.
  // Have to explicitly cast... :(
  m.def("call_method", static_cast<void(*)(const Cls&)>(&call_method));

  // For each conversion available:
  // Register base conversion(s).
  using To = Cls;
  using ToPtr = std::unique_ptr<To>;
  using From = Base<U, T>;
  auto from_tup = type_registry.GetPyTypes<U, T>();
  // Add Python converter function, but bind using Cls++ overloads via pybind.
  auto add_py_converter = [](BaseConverter* converter, py::function py_func) {
    // Add type information.
    using Func = std::function<ToPtr(const From&)>;
    auto cpp_func = py::cast<Func>(py_func);
    // Add using overload.
    converter->Add(cpp_func);
  };
  // Register function dispatch.
  auto key = py::make_tuple(type_tup, from_tup);
  tpl.attr("_add_py_converter_map")[key] = py::cpp_function(add_py_converter);
  // Add Python conversion.
  base
    .def(py::init<const From&>());
  // End: Scalar conversion.
}

PYBIND11_MODULE(_scalar_type, m) {
  m.doc() = "Simple check on scalar / template types";

  py::handle py_tpl = py::module::import("pymodule.tpl.py_tpl");
  py::handle tpl_cls = py_tpl.attr("Template");

  py::object tpl = tpl_cls("Base");
  m.attr("BaseTpl") = tpl;
  // Add instantiations and conversion mechanisms.
  tpl.attr("_add_py_converter_map") = py::dict();
  BaseTplInstantiation<int, double>(m, tpl);
  BaseTplInstantiation<double, int>(m, tpl);
  // Default instantiation.
  m.attr("Base") = tpl.attr("get_class")();
  // Register BaseConverter...
  py::class_<BaseConverter> base_converter(m, "BaseConverter");
  base_converter
    .def(py::init<>())
    .def(
      "Add",
      [tpl](BaseConverter* self,
             py::tuple params_to, py::tuple params_from,
             py::function py_converter) {
        // Assume we have canonical Python types.
        // Find our conversion function using these types.
        auto key = py::make_tuple(params_to, params_from);
        auto add_py_converter = tpl.attr("_add_py_converter_map")[key];
        // Now register the converter.
        add_py_converter(self, py_converter);
      });

  m.def("do_convert", &do_convert);
  m.def("take_ownership", &take_ownership);
}

}  // namespace scalar_type
