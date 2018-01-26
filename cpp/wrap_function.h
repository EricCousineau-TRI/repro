#include <functional>
#include <type_traits>

#include "cpp/name_trait.h"

namespace detail {

template <typename Func, typename Return, typename ... Args>
struct function_info {
  std::decay_t<Func> func;
};

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
auto infer_function_ptr() {
  return detail::remove_class_from_ptr(&Func::operator());
}

template <typename Func, typename T = void>
using enable_if_lambda_t =
    std::enable_if_t<std::integral_constant<
        bool, !std::is_function<std::decay_t<Func>>::value>::value, T>;

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
      std::forward<Func>(func),
      detail::infer_function_ptr<std::decay_t<Func>>());
}

// Nominal case.
template <template <typename...> class wrap_arg_tpl>
struct wrap_function_impl {
  template <typename T>
  struct wrap_arg : public wrap_arg_tpl<T> {};

  // Use `Extra` so that we can specialize within class scope.
  template <typename T, typename Extra>
  struct wrap_arg_t_impl {
    using type = decltype(wrap_arg<T>::wrap(std::declval<T>()));
  };

  template <typename Extra>
  struct wrap_arg_t_impl<void, Extra> {
    using type = void;
  };

  template <typename T>
  using wrap_arg_t = typename wrap_arg_t_impl<T, void>::type;

  template <typename Return>
  static constexpr bool enable_wrap_output =
      !std::is_same<Return, void>::value;

  template <typename Func, typename Return, typename ... Args>
  static auto run(function_info<Func, Return, Args...>&& info,
      std::enable_if_t<enable_wrap_output<Return>, void*> = {}) {
    auto func_wrapped =
        [func_f = std::forward<Func>(info.func)]
        (wrap_arg_t<Args>... args_wrapped) {
      return wrap_arg<Return>::wrap(
          func_f(wrap_arg<Args>::unwrap(
                  std::forward<wrap_arg_t<Args>>(args_wrapped))...));
    };
    return func_wrapped;
  }

  // Do not wrap output (or `Return` is void).
  template <typename Func, typename Return, typename ... Args>
  static auto run(function_info<Func, Return, Args...>&& info,
      std::enable_if_t<!enable_wrap_output<Return>, void*> = {}) {
    auto func_wrapped =
        [func_f = std::forward<Func>(info.func)]
        (wrap_arg_t<Args>... args_wrapped) {
      return func_f(wrap_arg<Args>::unwrap(
              std::forward<wrap_arg_t<Args>>(args_wrapped))...);
    };
    return func_wrapped;
  }

  // General case: Callbacks.
  template <typename Return, typename ... Args>
  struct wrap_arg<const std::function<Return (Args...)>&> {
    // Cannot use `auto`, because it is unable to mix lambdas.
    using Func = std::function<Return (Args...)>;
    using WrappedFunc = std::function<wrap_arg_t<Return> (wrap_arg_t<Args>...)>;

    static WrappedFunc wrap(const Func& func) {
      return wrap_function_impl::run(get_function_info(func));
    }

    template <typename Defer = Return>
    static Func unwrap(
        const WrappedFunc& func_wrapped,
        std::enable_if_t<!enable_wrap_output<Defer>, void*> = {}) {
      return [func_wrapped](Args... args) {
        func_wrapped(wrap_arg<Args>::wrap(std::forward<Args>(args))...);
      };
    }

    template <typename Defer = Return>
    static Func unwrap(
        const WrappedFunc& func_wrapped,
        std::enable_if_t<enable_wrap_output<Defer>, void*> = {}) {
      return [func_wrapped](Args... args) -> Return {
        return wrap_arg<Return>::unwrap(
            func_wrapped(wrap_arg<Args>::wrap(std::forward<Args>(args))...));
      };
    }
  };

  template <typename F>
  struct wrap_arg<std::function<F>>
      : public wrap_arg<const std::function<F>&> {};
};

}  // namespace detail

template <typename T>
struct wrap_arg_default {
  static T wrap(T arg) { return std::forward<T>(arg); }
  static T unwrap(T arg) { return std::forward<T>(arg); }
};

template <template <typename...> class wrap_arg_tpl, typename Func>
auto WrapFunction(Func&& func) {
  return detail::wrap_function_impl<wrap_arg_tpl>::run(
      detail::get_function_info(std::forward<Func>(func)));
}
