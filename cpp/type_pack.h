#pragma once

#include <utility>

template <size_t N, size_t K, typename T, typename ... Ts>
struct type_at_impl {
  using type = typename type_at_impl<N, K + 1, Ts...>::type;
};

template <size_t N, typename T, typename ... Ts>
struct type_at_impl<N, N, T, Ts...> {
  using type = T;
};

template <size_t N, typename ... Ts>
struct type_at {
  static_assert(N < sizeof...(Ts), "Invalid type index");
  using type = typename type_at_impl<N, 0, Ts...>::type;
};

template <typename ... Ts>
struct type_pack {
  template <template <typename...> class Tpl>
  using bind = Tpl<Ts...>;

  template <size_t N>
  using type_at = typename ::type_at<N, Ts...>::type;
};

// For decl-type only.
template <template <typename...> class Tpl, typename ... Ts>
Tpl<Ts...> type_bind(type_pack<Ts...>);

// Visitor setup.
struct visit_with_default {
  template <typename T, typename Visitor>
  inline static void run(Visitor&& visitor) {
    visitor(T{});
  }
};

template <typename T>
struct type_tag {
  using type = T;
};

template <template <typename> class Tag = type_tag>
struct visit_with_tag {
  template <typename T, typename Visitor>
  inline static void run(Visitor&& visitor) {
    visitor(Tag<T>{});
  }
};

template <typename VisitWith, typename Visitor>
struct type_visit_impl {
  template <typename T, bool execute>
  struct runner {
    inline static void run(Visitor&& visitor) {
      VisitWith::template run<T>(std::forward<Visitor>(visitor));
    }
  };
  template <typename T>
  struct runner<T, false> {
    inline static void run(Visitor&&) {}
  };
};

struct check_always_true {
  template <typename T>
  using check = std::true_type;
};

template <typename T>
using negation = std::integral_constant<bool, !T::value>;

template <typename T>
struct check_different_from {
  template <typename U>
  using check = negation<std::is_same<T, U>>;
};

using dummy_list = bool[];

template <class VisitWith = visit_with_default,
          typename Check = check_always_true,
          typename Visitor = void,
          typename ... Ts>
inline static void type_visit(
    Visitor&& visitor, type_pack<Ts...> pack = {}, Check check = {}) {
  (void)dummy_list{(
      type_visit_impl<VisitWith, Visitor>::
          template runner<Ts, Check::template check<Ts>::value>::
              run(std::forward<Visitor>(visitor)),
      true)...};
}


template <typename T>
struct type_pack_extract_impl {
  // Defer to show that this is a bad instantiation.
  static_assert(!std::is_same<T, T>::value, "Wrong template");
};

template <template <typename ... Ts> class Tpl, typename ... Ts>
struct type_pack_extract_impl<Tpl<Ts...>> {
  using type = type_pack<Ts...>;
};

template <typename T>
using type_pack_extract = typename type_pack_extract_impl<T>::type;

// Literal stuff.
template <typename TForm, typename T, T... Values>
auto transform(TForm = {}, std::integer_sequence<T, Values...> = {}) {
  return std::integer_sequence<T, TForm::template type<Values>::value...>{};
}

template <typename T, T x>
struct constant_add {
  template <T y>
  using type = std::integral_constant<T, x + y>;
};

template <typename T, T x>
struct constant_mult {
  template <T y>
  using type = std::integral_constant<T, x * y>;
};


template <typename T>
constexpr size_t type_hash() {
  return std::type_index(typeid(T)).hash_code();
}
