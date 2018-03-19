#include <pybind11/embed.h>

namespace py = pybind11;

int main(int argc, char** argv) {
  py::scoped_interpreter guard;

  py::list py_argv;
  for (int i = 0; i < argc; ++i)
    py_argv.append(argv[i]);
  py::module::import("sys").attr("argv") = py_argv;

  py::exec(R"""(
import os
import pytest
import sys

os.chdir("python/pybind11/tests")
print(os.getcwd())
pytest.main(args=sys.argv[1:])
)""");

  return 0;
}
