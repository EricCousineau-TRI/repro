#include <functional>
#include <iostream>
#include <type_traits>

#include "cpp/name_trait.h"

using namespace std;

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

template <typename Class, typename Return, typename ... Args>
auto remove_class_from_ptr(Return (Class::*)(Args...)) {
  return (Return (*)(Args...)){};
}

template <typename Class, typename Return, typename ... Args>
auto remove_class_from_ptr(Return (Class::*)(Args...) const) {
  return (Return (*)(Args...)){};
}

template <typename Func>
auto infer_function_ptr(Func* func = nullptr) {
  return detail::remove_class_from_ptr(&Func::operator());
}

template <typename Func>
using enable_if_lambda_t =
    std::enable_if_t<std::integral_constant<
        bool, !std::is_function<std::decay_t<Func>>::value>::value>;

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

template <typename Func, typename = detail::enable_if_lambda_t<Func>>
auto get_function_info(Func&& func) {
  return detail::infer_function_info(
      std::forward<Func>(func), detail::infer_function_ptr(&func));
}

namespace detail {

// Nominal case.
template <template <typename...> class wrap_arg>
struct wrap_impl {
  template <typename T>
  using wrap_arg_t = decltype(wrap_arg<T>::wrap(std::declval<T>()));

  template <typename Return>
  static constexpr bool enable_wrap_output =
      !std::is_same<Return, void>::value;

  template <typename Func, typename Return, typename ... Args>
  static auto run(function_info<Func, Return, Args...>&& info,
      std::enable_if_t<enable_wrap_output<Return>, void*> = {}) {
    auto func_wrapped =
        [func_f = std::forward<Func>(info.func)]
        (wrap_arg_t<Args>... args_wrapped) mutable {
      return wrap_arg<Return>::wrap(
          func_f(std::forward<Args>(
              wrap_arg<Args>::unwrap(
                  std::forward<wrap_arg_t<Args>>(args_wrapped)))...));
    };
    return func_wrapped;
  }

  // Do not wrap output (or `Return` is void).
  template <typename Func, typename Return, typename ... Args>
  static auto run(function_info<Func, Return, Args...>&& info,
      std::enable_if_t<!enable_wrap_output<Return>, void*> = {}) {
    auto func_wrapped =
        [func_f = std::forward<Func>(info.func)]
        (wrap_arg_t<Args>... args_wrapped) mutable {
      return func_f(std::forward<Args>(
          wrap_arg<Args>::unwrap(
              std::forward<wrap_arg_t<Args>>(args_wrapped)))...);
    };
    return func_wrapped;
  }
};

}  // namespace detail

// Base case: Pass though.
template <typename T, typename = void>
struct ensure_ptr {
  static T unwrap(T arg) { return std::forward<T>(arg); }
  static T wrap(T arg) { return std::forward<T>(arg); }
};

template <typename T>
struct ensure_ptr<const T*> {
  static const T* wrap(const T* arg) {
    cout << "<const T*> wrap: " << nice_type_name<const T*>() << endl;
    return arg;
  }
  static const T* unwrap(const T* arg) {
    cout << "<const T*> unwrap: " << nice_type_name<const T*>() << endl;
    return arg;
  }
};

template <typename T>
struct ensure_ptr<const T&> {
  static const T* wrap(const T& arg) {
    cout << "<const T&> wrap: " << nice_type_name<const T&>() << endl;
    return &arg;
  }
  static const T& unwrap(const T* arg) {
    cout << "<const T&> unwrap: " << nice_type_name<const T&>() << endl;
    return *arg;
  }
};

// Reference case: Convert to pointer.
template <typename T>
struct ensure_ptr<T&> {
  static T* wrap(T& arg) {
    cout << "<T&> wrap: " << nice_type_name<T&>() << endl;
    return &arg;
  }
  static T& unwrap(T* arg) {
    cout << "<T&> unwrap: " << nice_type_name<T&>() << endl;
    return *arg;
  }
};

template <typename Func>
auto EnsurePtr(Func&& func);

template <typename T>
using ensure_ptr_t =
    typename detail::wrap_impl<ensure_ptr>::template wrap_arg_t<T>;

template <typename Func>
struct ensure_ptr<Func, detail::enable_if_lambda_t<Func>> {
  using PFunc = std::decay_t<Func>*;
  static auto wrap(Func func) {
    return EnsurePtr(std::forward<Func>(func));
  }

  template <typename Wrapped>
  static auto unwrap(Wrapped&& func_wrapped) {
    return unwrap_impl(
        std::forward<Wrapped>(func_wrapped),
        detail::infer_function_ptr(PFunc{}));
  }

  template <typename Wrapped, typename ... Args>
  static auto unwrap_impl(Wrapped&& func_wrapped, void (*infer)(Args...)) {
    (void)infer;
    return [func_wrapped](Args... args) {
      func_wrapped(ensure_ptr<Args>::wrap(std::forward<Args>(args))...);
    };
  }
};

template <typename Func>
auto EnsurePtr(Func&& func) {
  return detail::wrap_impl<ensure_ptr>::run(
      get_function_info(std::forward<Func>(func)));
}

struct MoveOnlyValue {
  MoveOnlyValue() = default;
  MoveOnlyValue(const MoveOnlyValue&) = delete;
  MoveOnlyValue(MoveOnlyValue&&) = default;
  int value{};
};

void Func_1(int value) {}
int* Func_2(int& value) { value += 1; return &value; }
const int& Func_3(const int& value) { return value; }
void Func_4(MoveOnlyValue value) {}
void Func_5(const int* value) {}

void Func_6(int& value, std::function<void (int&)> callback) {
  callback(value);
}

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

#define CHECK(expr) EVAL(expr); cout << "v.value = " << v.value << endl << endl

int main() {
  MoveOnlyValue v{10};
  CHECK(EnsurePtr(Func_1)(v.value));
  CHECK(cout << *EnsurePtr(Func_2)(&v.value));
  CHECK(cout << *EnsurePtr(Func_3)(&v.value));
  CHECK(EnsurePtr(Func_4)(MoveOnlyValue{}));
  CHECK(EnsurePtr(Func_5)(&v.value));

  CHECK(EnsurePtr(MyClass::Func)(MoveOnlyValue{}));
  MyClass c;
  const MyClass& c_const{c};
  CHECK(EnsurePtr(&MyClass::Method)(&c, &v));
  CHECK(EnsurePtr(&MyClass::Method_2)(&c_const, &v));

  MoveOnlyFunctor f;
  CHECK(EnsurePtr(std::move(f))(&v));
  ConstFunctor g;
  CHECK(EnsurePtr(g)(&v));
  const ConstFunctor& g_const{g};
  CHECK(EnsurePtr(g_const)(&v));

  // // Callback.
  // auto void_ref = [](int& value) {
  //   value += 100;
  // };
  // CHECK(EnsurePtr(Func_6)(&v.value, EnsurePtr(void_ref)));

  return 0;
}
