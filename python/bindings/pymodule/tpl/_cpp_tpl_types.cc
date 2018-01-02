#include <pybind11/pybind11.h>

#include "python/bindings/pymodule/tpl/cpp_tpl_types.h"

PYBIND11_MODULE(_cpp_tpl_types, m) {
  py::class_<TypeRegistry> type_registry_cls(m, "_TypeRegistry");
  type_registry_cls
    .def(py::init<>())
    .def("GetPyTypeCanonical", &TypeRegistry::GetPyTypeCanonical)
    .def("GetPyTypesCanonical", &TypeRegistry::GetPyTypesCanonical)
    .def("GetName", [](const TypeRegistry *self, py::handle arg1) {
      return self->GetName(arg1);
    })
    // .def("GetName", py::overload_cast<py::handle>(&TypeRegistry::GetName))
    .def("GetNames", [](const TypeRegistry *self, py::tuple arg1) {
      return self->GetNames(arg1);
    });
  // Create instance.
  m.attr("type_registry") = type_registry_cls();
}
