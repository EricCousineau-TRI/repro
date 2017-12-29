#include <pybind11/pybind11.h>

#include "python/bindings/pymodule/tpl/cpp_tpl_types.h"

PYBIND11_MODULE(_cpp_tpl_types, m) {
  py::class_<TypeRegistry> type_registry_cls(m, "_TypeRegistry");
  type_registry_cls
    .def(py::init<>())
    .def("GetPyTypeCanonical", &TypeRegistry::GetPyTypeCanonical)
    .def("GetPyTypesCanonical", &TypeRegistry::GetPyTypesCanonical)
    .def("GetCppName", &TypeRegistry::GetCppName)
    .def("GetCppNames", &TypeRegistry::GetCppNames);
  // Create instance.
  m.attr("type_registry") = type_registry_cls();
}
