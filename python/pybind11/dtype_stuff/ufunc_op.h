#pragma once

#include "ufunc_utility.h"

// Could increase param count?
#define CHECK_EXPR(struct_name, name, expr, lambda_expr) \
    template <typename A_, typename B = A_> \
    struct struct_name { \
      template <typename A = A_> \
      static std::true_type check(decltype(expr)*); \
      template <typename> \
      static std::false_type check(...); \
      static constexpr bool value = decltype(check<A_>(nullptr))::value; \
      template <typename A = A_> \
      static auto get_lambda() { return lambda_expr; } \
      static const char* get_name() { return name; } \
    };

// https://docs.scipy.org/doc/numpy/reference/routines.math.html
CHECK_EXPR(check_add, "add", A{} + B{},
           [](const A& a, const B& b) { return a + b; });
CHECK_EXPR(check_negative, "negative", -A{},
           [](const A& a) { return -a; });
CHECK_EXPR(check_multiply, "multiply", A{} * B{},
           [](const A& a, const B& b) { return a * b; });
CHECK_EXPR(check_divide, "divide", A{} / B{},
           [](const A& a, const B& b) { return a / b; });
// TODO(eric.cousineau): Figger out non-operator things...
CHECK_EXPR(check_power, "power", pow(A{}, B{}),
           [](const A& a, const B& b) { return pow(a, b); });
CHECK_EXPR(check_subtract, "subtract", A{} - B{},
           [](const A& a, const B& b) { return a - b; });
// https://docs.scipy.org/doc/numpy/reference/routines.logic.html
CHECK_EXPR(check_greater, "greater", A{} > B{},
           [](const A& a, const B& b) { return a > b; });
CHECK_EXPR(check_greater_equal, "greater_equal", A{} >= B{},
           [](const A& a, const B& b) { return a >= b; });
CHECK_EXPR(check_less, "less", A{} < B{},
           [](const A& a, const B& b) { return a < b; });
CHECK_EXPR(check_less_equal, "less_equal", A{} <= B{},
           [](const A& a, const B& b) { return a <= b; });
CHECK_EXPR(check_equal, "equal", A{} == B{},
           [](const A& a, const B& b) { return a == b; });
CHECK_EXPR(check_not_equal, "not_equal", A{} != B{},
           [](const A& a, const B& b) { return a != b; });
// Casting.
CHECK_EXPR(check_cast, "", static_cast<B>(A{}),
           [](const A& a) { return static_cast<B>(a); });

template <typename Result, typename Func>
void run_if(Func&& func) {
  using Pack = type_pack<Result>;
  type_visit_impl<visit_with_default, Func>::
      template runner<Pack, Result::value>::run(func);
}

PyUFuncObject* get_py_ufunc(const char* name) {
  py::module numpy = py::module::import("numpy");
  return (PyUFuncObject*)numpy.attr(name).ptr();
}

template <template <typename...> class Check, typename Class, typename ... Args>
void maybe_ufunc(type_pack<Args...> = {}) {
  using Result = typename type_pack<Args...>::template bind<Check>;
  constexpr int N = sizeof...(Args);
  auto defer = [](auto) {
    RegisterUFunc<Class>(
        get_py_ufunc(Result::get_name()), Result::get_lambda(), const_int<N>{});
  };
  run_if<Result>(defer);
}

template <typename From, typename To, typename Func>
void add_cast(Func&& func, type_pack<From, To> = {}) {
  auto* from = PyArray_DescrFromType(dtype_num<From>());
  int to = dtype_num<To>();
  static auto cast_lambda = func;
  auto cast_func = [](
        void* from_, void* to_, npy_intp n,
        void* fromarr, void* toarr) {
      const From* from = (From*)from_;
      To* to = (To*)to_;
      for (npy_intp i = 0; i < n; i++)
          to[i] = cast_lambda(from[i]);
  };
  PY_ASSERT_EX(
      PyArray_RegisterCastFunc(from, to, cast_func) >= 0,
      "Cannot register cast");
  PY_ASSERT_EX(
      PyArray_RegisterCanCast(from, to, NPY_NOSCALAR) >= 0,
      "Cannot register castability");
}

template <typename From, typename To>
void maybe_cast(type_pack<From, To> = {}) {
  using Result = check_cast<From, To>;
  auto defer = [](auto) {
    add_cast<From, To>(Result::get_lambda());
  };
  run_if<Result>(defer);
}
