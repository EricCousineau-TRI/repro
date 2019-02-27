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
        self.check_poly(obj.get_poly_cptr())
        self.check_poly(obj.get_poly_ref())
        self.check_poly(obj.get_poly_cref())
        # Ownership transferred from C++ to Python.
        self.check_poly(mut.make_poly_smart_ptr())

    def test_py_to_cpp(self):
        poly = vtk.vtkPolyData()

        def check_take(f):
            self.assertEqual(f(poly), "vtkPolyData")

        check_take(mut.take_poly_smart_ptr)
        check_take(mut.take_poly_ptr)
        check_take(mut.take_poly_cptr)
        check_take(mut.take_poly_ref)
        check_take(mut.take_poly_cref)


if __name__ == "__main__":
    sys.stdout = sys.stderr
    tracer = trace.Trace(
        trace=1, count=0, ignoredirs=["/usr", sys.prefix])
    tracer.runfunc(unittest.main)
