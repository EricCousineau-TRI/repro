#include <iostream>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <pybind11/embed.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

using std::cout;
using std::endl;

int main() {
  py::scoped_interpreter guard;

  using constants = py::detail::npy_api::constants;

  py::module np = py::module::import("numpy");
  py::globals()["np"] = np;

  // Get `pybind11`s interpretation:
  cout << py::detail::npy_format_descriptor<int64_t>::dtype()
          .attr("num").template cast<int>() << endl;
  cout << constants::NPY_LONGLONG_ << endl;

  // Get NumPy's interpretation (what should be used):
  cout << constants::NPY_LONG_ << endl;  // What we should be using.
  cout << NPY_LONG << endl;
  cout << py::eval("np.dtype(np.int64).num").cast<int>() << endl;

  return 0;
}
