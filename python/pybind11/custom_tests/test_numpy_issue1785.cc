// https://github.com/pybind/pybind11/issues/1785
#include <Eigen/Dense>

#include <pybind11/embed.h>
#include <pybind11/eval.h>
// Using Eigen, 'cause I don't want to worry about ownership with capsules
// or buffers.
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using float128 = long double;
static_assert(sizeof(float128) == 16, "Bad size");

template <typename T, int Dim>
using Vector = Eigen::Matrix<T, Dim, 1>;

void init_module(py::module m) {
  m.def("make_array", []() {
    return Vector<float128, 3>(1, 2, 3);
  });
  m.def("sum_array", [](Vector<float128, -1> x) {
    // Return as 1x1 - I don't know how to easily return NumPy scalars...
    return Vector<float128, 1>(x.array().sum());
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

def main():
    x = np.array([1, 2], dtype=np.float128)
    print(repr(m.sum_array(x)))
    print(repr(m.make_array()))

main()
)""");

  py::print("[ Done ]");

  return 0;
}

/* Output:

[ Eval ]
array([ 3.0], dtype=float128)
array([ 1.0,  2.0,  3.0], dtype=float128)
[ Done ]

*/
