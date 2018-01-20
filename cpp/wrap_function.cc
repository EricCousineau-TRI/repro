#include <functional>
#include <iostream>

#include "cpp/name_trait.h"

using namespace std;

namespace detail {

template <typename T>
struct wrap_arg {
  using type_in = T;
  static T&& unwrap(type_in&& arg) { return arg; }
  static type_in&& wrap(T&& arg) { return arg; }
};

template <>
struct wrap_arg<void> {
  using type_in = void;
  static void unwrap() {}
  static void wrap() {}
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

}  // namespace detail

// TODO(eric.cousineau): Make this lightweight, if the callback does not have
// capture?
template <typename Return, typename ... Args, typename Func>
auto WrapFunctionImpl(Func&& func) {
  using detail::wrap_arg;
  using detail::wrap_arg_in_t;
  auto func_wrapped =
      [func_f = std::forward<Func>(func)](wrap_arg_in_t<Args>... args) {
    return wrap_arg<Return>::wrap(
        func_f(std::forward<Args>(
            wrap_arg<Args>::unwrap(
                std::forward<wrap_arg_in_t<Args>>(args)))...));
  };
  return func_wrapped;
}

template <typename Return, typename Class, typename ... Args>
auto Wrap(Return (Class::*method)(Args...)) {
  auto func = [method](Class* self, Args... args) {
    return (self->*method)(std::forward<Args>(args)...);
  };
  return WrapFunctionImpl<Return, Args...>(func);
}

template <typename Return, typename ... Args>
auto Wrap(Return (*func)(Args...)) {
  return WrapFunctionImpl<Return, Args...>(func);
}

template <typename Return, typename ... Args>
auto Wrap(std::function<Return (Args...)> func) {
  return WrapFunctionImpl<Return, Args...>(std::move(func));
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
  void Method(MoveOnlyValue& value) { value.value += 3; }
};

struct MoveOnlyFunctor {
  MoveOnlyFunctor(const MoveOnlyFunctor&) = delete;
  MoveOnlyFunctor(MoveOnlyFunctor&&) = default;

  void operator()(MoveOnlyValue& value) { value.value += 4; }
};

int main() {
  MoveOnlyValue v{10};

  auto w1 = Wrap(&Func_1);
  cout <<
    PRINT(w1(v.value));

  return 0;
}
