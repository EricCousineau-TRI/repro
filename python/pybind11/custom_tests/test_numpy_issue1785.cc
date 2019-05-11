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

namespace pybind11 { namespace detail {

template <>
struct type_caster<float128> {
  PYBIND11_TYPE_CASTER(float128, _("float128"));
  using arr_t = array_t<float128>;

  bool load(handle src, bool convert) {
    // Taken from Eigen casters.
    if (!convert && !isinstance<arr_t>(src))
      return false;
    arr_t tmp = arr_t::ensure(src);
    if (tmp && tmp.size() == 1 && tmp.ndim() == 0) {
      this->value = *tmp.data();
      return true;
    }
    return false;
  }

  static handle cast(float128 src, return_value_policy, handle) {
    arr_t tmp({1});
    tmp.mutable_at(0) = src;
    tmp.resize({});
    // You could also just return the array if you want a scalar array.
    object scalar = tmp[py::tuple()];
    return scalar.release();
  }
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

def main():
    info(m.incr_scalar(1.))
    x = np.array([1, 2], dtype=np.float128)
    info(m.sum_array(x))
    info(m.make_array())

main()
)""");

  py::print("[ Done ]");

  return 0;
}

/* Output:

2.0 <class 'numpy.float128'>
3.0 <class 'numpy.float128'>
array([ 1.0,  2.0,  3.0], dtype=float128) <class 'numpy.ndarray'>

*/
