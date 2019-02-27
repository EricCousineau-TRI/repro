#include <pybind11/pybind11.h>
#include <vtkPolyData.h>

#include "third_party/vtk_pybind_conversion.h"

PYBIND11_MODULE(vtk_pybind, m) {
  m.def("make_polydata", []() {
    return vtkSmartPointer<vtkPolyData>::New().GetPointer();
  });
}
