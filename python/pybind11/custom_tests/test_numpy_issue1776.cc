// https://github.com/pybind/pybind11/issues/1776
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "half.h"
#include "pybind11_numpy_scalar.h"

namespace py = pybind11;

using float16 = half_float::half;
static_assert(sizeof(float16) == 2, "Bad size");

namespace pybind11 { namespace detail {

// Similar to enums in `pybind11/numpy.h`. Determined by doing:
// python3 -c 'import numpy as np; print(np.dtype(np.float16).num)'
constexpr int NPY_FLOAT16 = 23;

// Kinda following: https://github.com/pybind/pybind11/blob/9bb3313162c0b856125e481ceece9d8faa567716/include/pybind11/numpy.h#L1000
template <>
struct npy_format_descriptor<float16> {
  static constexpr auto name = _("float16");
  static pybind11::dtype dtype() {
    handle ptr = npy_api::get().PyArray_DescrFromType_(NPY_FLOAT16);
    return reinterpret_borrow<pybind11::dtype>(ptr);
  }
};

template <>
struct type_caster<float16> : npy_scalar_caster<float16> {
  static constexpr auto name = _("float16");
};

}}  // namespace pybind11::detail

void init_module(py::module m) {
  m.def("make_array", []() {
    py::array_t<float16> x({2});
    x.mutable_at(0) = float16{1.};
    x.mutable_at(1) = float16{10.};
    return x;
  });
  m.def("make_scalar", []() { return float16{2.}; });
}

int main(int, char**) {
  py::scoped_interpreter guard{};

  py::module m("test_module");
  init_module(m);
  py::globals()["m"] = m;

  py::print("[ Eval ]");
  // See `test_numpy_issue1785` for more expansive tests for
  // `npy_scalar_caster`.
  py::exec(R"""(
import numpy as np

def info(x):
    print(repr(x), type(x))

def main():
    info(m.make_array())
    info(m.make_scalar())

main()
)""");

  py::print("[ Done ]");

  return 0;
}

/* Output:

array([  1.,  10.], dtype=float16) <class 'numpy.ndarray'>
2.0 <class 'numpy.float16'>

*/
