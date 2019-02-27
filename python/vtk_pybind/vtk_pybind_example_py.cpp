#include <pybind11/pybind11.h>
#include <vtkPolyData.h>

#include "vtk_pybind.h"

namespace py = pybind11;
using rvp = py::return_value_policy;

struct CppOwned {
  vtkNew<vtkPolyData> poly;

  vtkPolyData* get_poly_ptr() { return poly.GetPointer(); }
  const vtkPolyData* get_poly_cptr() { return poly.GetPointer(); }
  vtkPolyData& get_poly_ref() { return *poly; }
  const vtkPolyData& get_poly_cref() { return *poly; }
};

PYBIND11_MODULE(vtk_pybind_example, m) {
  // Use a class to control the lifetime of a VTK object.
  py::class_<CppOwned>(m, "CppOwned")
    .def(py::init())
    .def_readonly("poly", &CppOwned::poly)  // test access to `vtkNew<T>`
    .def("get_poly_ptr", &CppOwned::get_poly_ptr)
    .def("get_poly_cptr", &CppOwned::get_poly_cptr)
    .def("get_poly_ref", &CppOwned::get_poly_ref, rvp::reference)
    .def("get_poly_cref", &CppOwned::get_poly_cref, rvp::reference);

  // Create in C++, pass to Python.
  m.def("make_poly_smart_ptr", []() {
    return vtkSmartPointer<vtkPolyData>::New();
  });

  // Take from Python.
  m.def("take_poly_smart_ptr", [](vtkSmartPointer<vtkPolyData> poly) {
    return poly->GetClassName();
  });
  m.def("take_poly_ptr", [](vtkPolyData* poly) {
    return poly->GetClassName();
  });
  m.def("take_poly_cptr", [](const vtkPolyData* poly) {
    return poly->GetClassName();
  });
  m.def("take_poly_ref", [](vtkPolyData& poly) {
    return poly.GetClassName();
  });
  m.def("take_poly_cref", [](const vtkPolyData& poly) {
    return poly.GetClassName();
  });
}
