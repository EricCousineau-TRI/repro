// For: https://github.com/pybind/pybind11/pull/2000

#include <chrono>
#include <thread>

#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using rvp = py::return_value_policy;

void init_module(py::module m) {
  m.def("stuff", []() -> void {
    while (true) {
      if (PyErr_CheckSignals() != 0) {
        throw py::error_already_set();
      }
      using namespace std::chrono_literals;
      std::this_thread::sleep_for(50ms);
    }
  });
}

int main(int, char**) {
  py::scoped_interpreter guard{};

  py::module m("test_module");
  init_module(m);
  py::globals()["m"] = m;

  py::print("[ Eval ]");
  py::exec(R"""(
try:
  print("Press Ctrl+C to stop")
  m.stuff()
  assert False
except KeyboardInterrupt:
  print("[ Done ]")
)""");

  return 0;
}
