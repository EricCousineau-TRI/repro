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

struct always_true {
  template <typename T>
  using check = std::true_type;
};

template <typename T>
using negation = std::integral_constant<bool, !T::value>;

template <typename T>
struct is_different_from {
  template <typename U>
  using check = negation<std::is_same<T, U>>;
};

template <typename ... Ts>
struct type_pack {
  template <template <typename...> class Tpl>
  using bind = Tpl<Ts...>;

  template <size_t N>
  using type = typename type_at<N, Ts...>::type;
};

// For when `visit` is used with a Pack (and doesn't need a tag).
template <typename T>
using use_type = T;

template <typename T>
struct type_tag {
  using type = T;
};

template <template <typename> class Tag, typename Visitor>
struct types_visit_impl {
  template <typename T, bool execute>
  struct runner {
    inline static void run(Visitor&& visitor) {
      visitor(Tag<T>{});
    }
  };
  template <typename T>
  struct runner<T, false> {
    inline static void run(Visitor&&) {}
  };
};

using dummy_list = bool[];

template <template <typename> class Tag = use_type,
          typename Check = void,
          typename Visitor = void,
          typename ... Ts>
inline static void type_pack_visit(
    Visitor&& visitor, type_pack<Ts...> pack = {}, Check check = {}) {
  (void)dummy_list{(
      types_visit_impl<Tag, Visitor>::
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

  template <template <typename...> class TplIn>
  using type_constrained =
      typename std::conditional<
          std::is_same<TplIn<Ts...>, Tpl<Ts...>>::value, 
            type,
            std::false_type
      >::type;
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

// Unused.
template <typename T, template <typename...> class Tpl>
using type_pack_extract_constrained =
    typename type_pack_extract_impl<T>::template type_constrained<Tpl>;
