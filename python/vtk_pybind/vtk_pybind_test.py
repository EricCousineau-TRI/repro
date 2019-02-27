import sys
import trace

import vtk

import vtk_pybind as m


def info(x):
    return "<{} at {}>".format(type(x).__name__, id(x))


def check_poly(x):
    print(info(x))
    assert isinstance(x, vtk.vtkPolyData)


def main():
    print("Hello")
    poly_py = vtk.vtkPolyData()
    check_poly(poly_py)
    poly_cc_owned = m.Test().get_poly()
    check_poly(poly_cc_owned)
    poly_cc_transfer = m.make_poly()  # Segfault...
    check_poly(poly_cc_transfer)


if __name__ == "__main__":
    sys.stdout = sys.stderr
    tracer = trace.Trace(
        trace=1, count=0, ignoredirs=["/usr", sys.prefix])
    tracer.runfunc(main)
