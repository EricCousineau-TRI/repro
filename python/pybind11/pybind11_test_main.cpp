#include <pybind11/embed.h>

namespace py = pybind11;

int main(int argc, char** argv) {
  py::scoped_interpreter guard;

  py::list py_argv;
  for (int i = 0; i < argc; ++i)
    py_argv.append(argv[i]);
  py_argv.append("-s");
  py::module::import("sys").attr("argv") = py_argv;

  py::globals()["__file__"] = py_argv[0];

  py::exec(R"""(
import os
import pytest
import sys
import trace

sys.stdout = sys.stderr

# For debug in CLion, PWD sucks. Override it.
os.chdir(sys.argv[0] + ".runfiles/repro")
# Change directory, and execute.
os.chdir("python/pybind11/tests")
tracer = trace.Trace(trace=1, count=0, ignoredirs=sys.path)
tracer.run('pytest.main(args=sys.argv[1:])')
)""");

  return 0;
}
