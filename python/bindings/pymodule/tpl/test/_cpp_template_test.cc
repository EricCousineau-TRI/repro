#include <iostream>
#include <string>

#include <pybind11/pybind11.h>

#include "cpp/name_trait.h"
#include "python/bindings/pymodule/tpl/cpp_template.h"


namespace cpp_template_test {

struct SimpleType {};

template <typename T>
void template_type() {
  std::cout << "template_type: " << nice_type_name<T>() << std::endl;
}

template <typename... Ts>
void template_list() {
  std::cout << "template_list: " << std::endl;
  for (std::string name : {nice_type_name<Ts>()...}) {
    std::cout << "- " << name << std::endl;
  }
}

template <typename... Ts>
class SimpleTemplate {
 public:
  int size() const {
    return type_pack<Ts...>::size;
  }

  template <typename U>
  void check() const {
    using Check = check_different_from<U>;
    std::cout << "check: ";
    for (bool value : {Check::template check<Ts>::value...}) {
      std::cout << value << " ";
    }
    std::cout << std::endl;
  }
};

template <bool Value>
void template_bool() {
  std::cout << "template_bool: " << Value << std::endl;
}

template <int Value>
void template_int() {
  std::cout << "template_int: " << Value << std::endl;
}

template <typename T, T ... Values>
using type_pack_literals =
    type_pack<type_pack<std::integral_constant<T, Values>>...>;

template <typename T, T ... Values>
using type_pack_literals_raw =
    type_pack<std::integral_constant<T, Values>...>;

}  // namespace cpp_template_test

PYBIND11_MODULE(_cpp_template_test, m) {
  using namespace cpp_template_test;

  m.doc() = "C++ Template Test";

  // Custom type used in templates.
  py::class_<SimpleType>(m, "SimpleType");

  // Types - Manual.
  AddTemplateFunction<int>(
      m, "template_type", &template_type<int>);
  AddTemplateFunction<double>(
      m, "template_type", &template_type<double>);
  AddTemplateFunction<SimpleType>(
      m, "template_type", &template_type<SimpleType>);

  // - Lists
  AddTemplateFunction<int>(
      m, "template_list", &template_list<int>);
  AddTemplateFunction<int, double>(
      m, "template_list", &template_list<int, double>);
  AddTemplateFunction<int, double, SimpleType>(
      m, "template_list", &template_list<int, double, SimpleType>);

  // - Class w/ looping.
  {
    auto inst = [&m](auto param) {
      using Param = decltype(param);
      using SimpleTemplateT = typename Param::template bind<SimpleTemplate>;
      // N.B. This name will be overwritten by `AddTemplateClass`.
      py::class_<SimpleTemplateT> py_class(
          m, TemplateClassName<SimpleTemplateT>().c_str());
      py_class
        .def(py::init<>())
        .def("size", &SimpleTemplateT::size);
      AddTemplateMethod<double>(
          py_class, "check", &SimpleTemplateT::template check<double>);
      AddTemplateClass(
          m, "SimpleTemplateTpl", py_class, "SimpleTemplate", param);
    };
    using ParamList = type_pack<
        type_pack<int>,
        type_pack<int, double, SimpleType>>;
    type_visit(inst, ParamList{});
  }

  // Literals.
  {
    // Manual - must wrap with `integral_constant`, since that is what is
    // registered.
    AddTemplateFunction<std::integral_constant<int, 0>>(
        m, "template_int", &template_int<0>);

    // Looping, raw.
    auto inst = [&m](auto tag) {
      using Tag = decltype(tag);
      constexpr int Value = Tag::value;
      AddTemplateFunction<Tag>(
          m, "template_int", &template_int<Value>);
    };
    type_visit(inst, type_pack_literals_raw<int, 1, 2, 5>{});
  }

  {
    // Looping, Type packs.
    auto inst = [&m](auto param) {
      constexpr bool Value = decltype(param)::template type_at<0>::value;
      // N.B. Use of `param` argument, no template specification.
      AddTemplateFunction(
          m, "template_bool", &template_bool<Value>, param);
    };
    type_visit(inst, type_pack_literals<bool, false, true>{});
  }
}
