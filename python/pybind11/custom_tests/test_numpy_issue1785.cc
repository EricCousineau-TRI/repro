// https://github.com/pybind/pybind11/issues/1785
#include <Eigen/Dense>

#include <pybind11/embed.h>
#include <pybind11/eval.h>
// Using Eigen, 'cause I don't want to worry about ownership with capsules
// or buffers.
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "pybind11_numpy_scalar.h"

namespace py = pybind11;

using float128 = long double;
static_assert(sizeof(float128) == 16, "Bad size");

template <typename T, int Dim>
using Vector = Eigen::Matrix<T, Dim, 1>;

namespace pybind11 { namespace detail {

template <>
struct type_caster<float128> : npy_scalar_caster<float128> {
  static constexpr auto name = _("float128");
};

}}  // namespace pybind11::detail

void init_module(py::module m) {
  m.def("make_array", []() {
    return Vector<float128, 3>(1, 2, 3);
  });
  m.def("sum_array", [](Vector<float128, -1> x) {
    return x.array().sum();
  });
  m.def("incr_scalar", [](float128 x) { return x + 1.; });

  // Check overloads
  m.def("overload", [](float128 x) { return "float128"; });
  m.def("overload", [](Vector<float128, 1> x) { return "ndarray[float128, 1]"; });
  m.def("overload", [](int x) { return "int"; });  // Put at end, due to things like pybind11#1392 ???
}

int main(int, char**) {
  py::scoped_interpreter guard{};

  py::module m("test_module");
  init_module(m);
  py::globals()["m"] = m;

  py::print("[ Eval ]");
  py::exec(R"""(
import numpy as np

def info(x):
    print(repr(x), type(x))

def check(a, b):
    assert a == b, (a, b)

def main():
    info(m.incr_scalar(1.))
    x = np.array([1, 2, 3], dtype=np.float128)
    info(m.sum_array(x))
    info(m.make_array())

    check(m.overload(1), "int")
    check(m.overload(np.float128(1.)), "float128")
    check(m.overload(np.array([1], np.float128)), "ndarray[float128, 1]")

main()
)""");

  py::print("[ Done ]");

  return 0;
}

/* Output:

2.0 <class 'numpy.float128'>
6.0 <class 'numpy.float128'>
array([ 1.0,  2.0,  3.0], dtype=float128) <class 'numpy.ndarray'>

*/
