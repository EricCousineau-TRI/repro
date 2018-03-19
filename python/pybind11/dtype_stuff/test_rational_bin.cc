#include <pybind11/embed.h>

namespace py = pybind11;

int main(int argc, char** argv) {
  py::scoped_interpreter guard;

  py::list py_argv;
  for (int i = 0; i < argc; ++i) py_argv.append(argv[i]);
  py::module::import("sys").attr("argv") = py_argv;

  py::globals()["__file__"] = "python/pybind11/dtype_stuff/test_rational_bin.py";
  py::exec(R"""(
import os, sys
os.chdir(sys.argv[0] + ".runfiles/repro")

sys.path.insert(0, os.path.dirname(__file__))
execfile(__file__)
)""");

  return 0;
}
