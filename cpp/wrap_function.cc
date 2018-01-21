#include <functional>
#include <iostream>
#include <type_traits>

#include "cpp/name_trait.h"

using namespace std;

namespace detail {

// Start: from pybind:
/// Strip the class from a method type
template <typename T> struct remove_class { };
template <typename C, typename R, typename... A> struct remove_class<R (C::*)(A...)> { typedef R type(A...); };
template <typename C, typename R, typename... A> struct remove_class<R (C::*)(A...) const> { typedef R type(A...); };
// Get function signature.
template <typename F> struct strip_function_object {
    using type = typename remove_class<decltype(&F::operator())>::type;
};
template <typename F>
using function_ptr_t =
    typename strip_function_object<std::decay_t<F>>::type*;
template <typename F>
using is_lambda = std::integral_constant<
    bool, !std::is_function<std::decay_t<F>>::value>;

// End: from pybind
}

template <typename Func, typename Return, typename ... Args>
struct function_info {
  std::decay_t<Func> func;
};

namespace detail {

template <typename Return, typename ... Args, typename Func>
auto infer_function_info(Func&& func, Return (*infer)(Args...) = nullptr) {
  (void)infer;
  return function_info<Func, Return, Args...>{std::forward<Func>(func)};
}

}  // namespace detail

template <typename Return, typename ... Args>
auto get_function_info(Return (*func)(Args...)) {
  return detail::infer_function_info<Return, Args...>(func);
}

template <typename Return, typename Class, typename ... Args>
auto get_function_info(Return (Class::*method)(Args...)) {
  auto func = [method](Class* self, Args... args) {
    return (self->*method)(std::forward<Args>(args)...);
  };
  return detail::infer_function_info<Return, Class*, Args...>(func);
}

template <typename Return, typename Class, typename ... Args>
auto get_function_info(Return (Class::*method)(Args...) const) {
  auto func = [method](const Class* self, Args... args) {
    return (self->*method)(std::forward<Args>(args)...);
  };
  return detail::infer_function_info<Return, const Class*, Args...>(func);
}

template <typename Func,
    typename = std::enable_if_t<detail::is_lambda<Func>::value>>
auto get_function_info(Func&& func) {
  return detail::infer_function_info(
      std::forward<Func>(func), detail::function_ptr_t<Func>{});
}

namespace detail {

template <typename T>
struct wrap_arg {
  using type_in = T;
  static T unwrap(T arg) { return std::forward<T>(arg); }
  static T wrap(T arg) { return std::forward<T>(arg); }
};

// Ensure that all reference types are passed as pointers.
template <typename T>
struct wrap_arg<T&> {
  using type_in = T*;
  static T& unwrap(type_in arg) { return *arg; }
  static type_in wrap(T& arg) { return &arg; }
};

template <typename T>
using wrap_arg_in_t = typename wrap_arg<T>::type_in;

// Nominal case.
template <typename Func, typename Return, typename ... Args>
auto WrapImpl(function_info<Func, Return, Args...>&& info) {
  auto func_wrapped =
      [func_f = std::forward<Func>(info.first)]
      (wrap_arg_in_t<Args>... args) mutable {
    return wrap_arg<Return>::wrap(
        func_f(std::forward<Args>(
            wrap_arg<Args>::unwrap(
                std::forward<wrap_arg_in_t<Args>>(args)))...));
  };
  return func_wrapped;
}

// Return `void` case (do not wrap output).
template <typename Func, typename ... Args>
auto WrapImpl(function_info<Func, void, Args...>&& info) {
  auto func_wrapped =
      [func_f = std::forward<Func>(info.func)]
      (wrap_arg_in_t<Args>... args) mutable {
    return func_f(std::forward<Args>(
        wrap_arg<Args>::unwrap(
            std::forward<wrap_arg_in_t<Args>>(args)))...);
  };
  return func_wrapped;
}

}  // namespace detail

template <typename Func>
auto Wrap(Func&& func) {
  return detail::WrapImpl(get_function_info(std::forward<Func>(func)));
}

struct MoveOnlyValue {
  MoveOnlyValue() = default;
  MoveOnlyValue(const MoveOnlyValue&) = delete;
  MoveOnlyValue(MoveOnlyValue&&) = default;
  int value{};
};

void Func_1(int value) {}
void Func_2(int& value) { value += 1; }
void Func_3(const int& value) { }
void Func_4(MoveOnlyValue value) {}

class MyClass {
 public:
  static void Func(MoveOnlyValue&& value) {}
  void Method(MoveOnlyValue& value) { value.value += 2; }
  void Method_2(MoveOnlyValue& value) const { value.value += 3; }
};

struct MoveOnlyFunctor {
  MoveOnlyFunctor() {}
  MoveOnlyFunctor(const MoveOnlyFunctor&) = delete;
  MoveOnlyFunctor(MoveOnlyFunctor&&) = default;
  // NOTE: Mutable operator().
  // Cannot overload operator(), as it's ambiguous.
  void operator()(MoveOnlyValue& value) { value.value += 4; }
};

struct ConstFunctor {
  void operator()(MoveOnlyValue& value) const { value.value += 5; }
};

#define CHECK(expr) EVAL(expr; cout << v.value);

int main() {
  MoveOnlyValue v{10};
  CHECK(Wrap(Func_1)(v.value));
  CHECK(Wrap(Func_2)(&v.value));
  CHECK(Wrap(Func_3)(&v.value));
  CHECK(Wrap(Func_4)(MoveOnlyValue{}));

  CHECK(Wrap(MyClass::Func)(MoveOnlyValue{}));
  MyClass c;
  const MyClass& c_const{c};
  CHECK(Wrap(&MyClass::Method)(&c, &v));
  CHECK(Wrap(&MyClass::Method_2)(&c_const, &v));

  MoveOnlyFunctor f;
  CHECK(Wrap(std::move(f))(&v));
  ConstFunctor g;
  CHECK(Wrap(g)(&v));
  const ConstFunctor& g_const{g};
  CHECK(Wrap(g_const)(&v));

  return 0;
}
