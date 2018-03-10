// taken from: numpy/core/src/umath/test_rational.c.src
// see ufuncobject.h


// Goal: Define functions that need no capture...


#include <array>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using py::detail::npy_format_descriptor;

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ufuncobject.h>

template <typename Type, typename ... Args>
void RegisterUFunc(
    PyUFuncObject* py_ufunc,
    PyUFuncGenericFunction func,
    void* data) {
  constexpr int N = sizeof...(Args);
  int dtype = npy_format_descriptor<Type>::value;
  int dtype_args[] = {npy_format_descriptor<Args>::value...};
  if (N != py_ufunc->nargs) {
    throw py::cast_error("bad stuff");
  }
  int result = PyUFunc_RegisterLoopForType(
      py_ufunc, dtype, func, dtype_args, 0);
  if (result < 0) throw py::cast_error("badder stuff");
}

template <typename Return, typename ... Args>
using Func = Return (*)(Args...);

template <
    typename Arg0, typename Arg1, typename Out,
    Func<Out, const Arg0&, const Arg1&> func>
struct BinaryUFunc {
  static void ufunc(
      char** args, npy_intp* dimensions, npy_intp* steps, void* data) {
    int step_0 = steps[0];
    int step_1 = steps[1];
    int step_out = steps[2];
    int n = *dimensions;
    char *in_0 = args[0], *in_1 = args[1], *out = args[2];
    for (int k = 0; k < n; k++) {
        *(Out*)out = func(*(Arg0*)in_0, *(Arg1*)in_1);
        in_0 += step_0;
        in_1 += step_1;
        out += step_out;
    }
  }
  template <typename Type = Arg0>
  static void Register(PyUFuncObject* py_ufunc) {
    RegisterUFunc<Type, Arg0, Arg1, Out>(py_ufunc, ufunc, nullptr);
  }
};
