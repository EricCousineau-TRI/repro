#include <pybind11/pybind11.h>
#include <vtkPolyData.h>

#include "third_party/vtk_pybind_conversion.h"

namespace py = pybind11;

struct Test {
  vtkNew<vtkPolyData> poly;
  vtkPolyData* get_poly() { return poly.GetPointer(); }
};

PYBIND11_MODULE(vtk_pybind, m) {
  // Use a class to control the lifetime of a VTK object.
  py::class_<Test>(m, "Test")
    .def(py::init())
    .def("get_poly", &Test::get_poly);

  // This segfaults because I (Eric) don't know how to transfer ownership from
  // a smart pointer to Python...
  m.def("make_poly", []() {
    return vtkSmartPointer<vtkPolyData>::New().GetPointer();
  });
}
