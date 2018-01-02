#pragma once

#include "cpp/simple_converter.h"
#include "python/bindings/pymodule/tpl/cpp_tpl.h"

template <typename Converter>
auto RegisterConverter(py::module m, py::object scope) {
  // Register converter. Name does not matter.
  py::class_<Converter> converter(m, "ConverterTmp");
  converter
    .def(py::init<>())
    .def(
      "Add",
      [scope](Converter* self,
             py::tuple to_param_py, py::tuple from_param_py,
             py::function py_converter) {
        // @pre `to_param_py` and `from_param_py` are canonical Python types.
        // Find conversion function using these types.
        py::tuple key = py::make_tuple(to_param_py, from_param_py);
        py::object add_py_converter = scope.attr("_add_py_converter_map")[key];
        // Add the casted converter.
        add_py_converter(self, py_converter);
      });
  scope.attr("Converter") = converter;
  scope.attr("_add_py_converter_map") = py::dict();
}

template <
    template <typename...> class Tpl, typename Converter,
    template <typename...> class Ptr = std::unique_ptr,
    typename ToParam = void, typename FromParamList = void,
    typename Check = is_different_from<ToParam>,
    // Use `void` here since these will be inferred, but allow the check to
    // have a default value.
    typename PyClass = void>
void RegisterConversions(
    PyClass& py_class, py::object scope,
    ToParam to_param = {}, FromParamList from_param_list = {}) {
  using To = typename ToParam::template bind<Tpl>;
  py::tuple to_param_py = get_py_types(to_param);
  auto add_conversion = [&](auto from_param) {
    // Register base conversion.
    using FromParam = decltype(from_param);
    using From = typename FromParam::template bind<Tpl>;
    py::tuple from_param_py = get_py_types(from_param);
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
    py::tuple key = py::make_tuple(to_param_py, from_param_py);
    scope.attr("_add_py_converter_map")[key] =
        py::cpp_function(add_py_converter);
    // Add Python conversion.
    py_class.def(py::init<const From&>());
  };
  FromParamList::template visit_if<Check, no_tag>(add_conversion);
}
