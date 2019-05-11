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
  using my_array = array_t<float128>;

  bool load(handle src, bool convert) {
    if (!convert && !isinstance<my_array>(src))
      return false;
    my_array tmp = my_array::ensure(src);
    if (tmp) {
      this->value = *tmp.data();
    }
    return false;
  }

  static handle cast(float128 src, return_value_policy, handle) {
    // // This causes a segfault:
    // using Wrapped = Vector<float128, 1>;
    // using wrap_caster = type_caster<Wrapped>;
    // return wrap_caster::cast(
    //     Wrapped(src), return_value_policy::move, handle()))
    //     .attr("reshape")(py::make_tuple());

    // This does not work (circular 'base')
    return array_t<float128>({1}, {1}, &src, handle());
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
