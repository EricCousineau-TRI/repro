// taken from: numpy/core/src/umath/test_rational.c.src
// see ufuncobject.h


// Goal: Define functions that need no capture...


#include <array>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "cpp/wrap_function.h"

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
  int dtype = npy_format_descriptor<Type>::dtype().num();
  int dtype_args[] = {npy_format_descriptor<Args>::dtype().num()...};
  if (N != py_ufunc->nargs) {
    throw py::cast_error("bad stuff");
  }
  int result = PyUFunc_RegisterLoopForType(
      py_ufunc, dtype, func, dtype_args, data);
  if (result < 0) throw py::cast_error("badder stuff");
}

template <typename Return, typename ... Args>
using Func = Return (*)(Args...);

template <typename T>
void* heapit(const T& x) { return new T(x); }
void* heapit(std::nullptr_t) { return nullptr; }

template <typename Type, typename Func>
void RegisterBinaryUFunc(PyUFuncObject* py_ufunc, Func func) {
    auto info = detail::infer_function_info(func);
    using Info = decltype(info);
    using Arg0 = std::decay_t<typename Info::Args::template type_at<0>>;
    using Arg1 = std::decay_t<typename Info::Args::template type_at<1>>;
    using Out = std::decay_t<typename Info::Return>;
    auto ufunc = [](char** args, npy_intp* dimensions, npy_intp* steps, void* data) {
        auto& func = *(Func*)data;
        int step_0 = steps[0];
        int step_1 = steps[1];
        int step_out = steps[2];
        int n = *dimensions;
        char *in_0 = args[0], *in_1 = args[1], *out = args[2];
        for (int k = 0; k < n; k++) {
            // TODO(eric.cousineau): Support pointers being changed.
            *(Out*)out = func(*(Arg0*)in_0, *(Arg1*)in_1);
            in_0 += step_0;
            in_1 += step_1;
            out += step_out;
        }
    };
    RegisterUFunc<Type, Arg0, Arg1, Out>(py_ufunc, ufunc, new Func(func));
};
