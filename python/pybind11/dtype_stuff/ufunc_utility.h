// taken from: numpy/core/src/umath/test_rational.c.src
// see ufuncobject.h

#include <array>
#include <string>

#include <pybind11/pybind11.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ufuncobject.h>

template <typename Type, typename ... Args, typename F>
void RegisterUFunc(
    PyUFuncObject* py_ufunc,
    PyUFuncGenericFunction func,
    void* data) {
  constexpr int N = sizeof...(Args);
  int dtype = npy_format_descriptor<Type>::value;
  int[] dtype_args = {npy_format_descriptor<Args...>::value};
  int nargs = In + 1;
  if (In + 1 != py_ufunc->nargs) {
    py::error("ufunc {} requires {} args, we have {}",
              name, py_ufunc->nargs, In + 1);
  }
  int result = PyUFunc_RegisterLoopForType(
      py_ufunc, dtype, func, dtype_args, 0);
  if (result < 0) py::error("crap");
}

template <typename Return, typename ... Args>
using Func = Return (*)(Args...);

template <typename Arg0, typename Arg1, typename Out>
auto RegisterBinaryUFunc(py::object py_ufunc, Func<Out, Arg0, Arg1> func) {
  PyUFuncGenericFunction lambda = [](
      char** args, npy_intp* dimensions, npy_intp* steps, void* data) {
    auto func = (Func<Out, Arg0, Arg1>)data;
    int step_0 = steps[0];
    int step_1 = steps[1];
    int step_out = steps[2];
    int n = *dimensions;
    char *in_0 = args[0], *in_1 = args[1], *out = args[2];
    for (int k = 0; k < n; k++) {
        *(Out*)out = func(*(Arg0)in_0, *(Arg1)in_1);
        in_0 += step_0;
        in_1 += step_1;
        out += step_out;
    }
  }
}
