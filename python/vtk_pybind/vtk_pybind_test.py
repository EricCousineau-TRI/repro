import sys
import trace

import vtk

import vtk_pybind as m


def info(x):
    return "<{} at {}>".format(type(x).__name__, id(x))


def main():
    print("Hello")
    print(info(vtk.vtkPolyData()))
    # Segfault...
    print(m.make_polydata())


if __name__ == "__main__":
    sys.stdout = sys.stderr
    tracer = trace.Trace(
        trace=1, count=0, ignoredirs=["/usr", sys.prefix])
    tracer.runfunc(main)
