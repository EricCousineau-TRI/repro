// From: https://github.com/RobotLocomotion/drake/issues/12633
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_module(py::module m) {
  m.def("pass_through", [](const Eigen::MatrixXd& A) {
    return A;
  });
  m.def("pass_through_ref", [](const Eigen::Ref<const Eigen::MatrixXd>& A) {
    return A;
  });
}

int main(int, char**) {
  py::scoped_interpreter guard{};

  py::module m("test_module");
  init_module(m);
  py::globals()["m"] = m;

  py::print("[ Eval ]");
  py::exec(R"""(
import numpy as np

def check(f):
    print(f"{f} - 0, 1")
    f(np.zeros((0, 1)))
    print(f"{f} - 0, 2")
    f(np.zeros((0, 2)))

check(m.pass_through)
check(m.pass_through_ref)
)""");

  py::print("[ Done ]");

  return 0;
}
