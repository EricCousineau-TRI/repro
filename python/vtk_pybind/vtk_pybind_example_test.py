import sys
import trace
import unittest

import vtk

# Module under test.
import vtk_pybind_example as mut


class TestVtkPybind(unittest.TestCase):
    def check_poly(self, poly):
        self.assertIsInstance(poly, vtk.vtkPolyData)

    def test_pure_py(self):
        self.check_poly(vtk.vtkPolyData())

    def test_cpp_to_py(self):
        # Owned by C++.
        obj = mut.CppOwned()
        self.check_poly(obj.poly)  # vtkNew<T>
        self.check_poly(obj.get_poly_ptr())
        # Ownership transferred from C++ to Python.
        self.check_poly(mut.make_poly_smart_ptr())


if __name__ == "__main__":
    sys.stdout = sys.stderr
    tracer = trace.Trace(
        trace=1, count=0, ignoredirs=["/usr", sys.prefix])
    tracer.runfunc(unittest.main)
