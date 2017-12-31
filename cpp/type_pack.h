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

template <typename T>
struct type_tag {
  using type = T;
};

template <template <typename> class Wrap, typename Visitor>
struct types_visit_wrap_impl {
  template <typename T, bool execute>
  struct runner {
    inline static void run(Visitor&& visitor) {
      visitor(Wrap<T>{});
    }
  };
  template <typename T>
  struct runner<T, false> {
    inline static void run(Visitor&&) {}
  };
};

using dummy_list = bool[];

// For when `visit` is used with a Pack (and doesn't need a tag).
template <typename T>
using no_tag = T;

template <typename ... Ts>
struct type_pack {
  template <template <typename...> class Tpl>
  using bind = Tpl<Ts...>;

  template <size_t N>
  using type = typename type_at<N, Ts...>::type;

  template <template <typename> class Wrap = type_tag, typename Visitor = void>
  inline static void visit(Visitor&& visitor) {
    (void)dummy_list{(
      visitor(Wrap<Ts>{}), true)...};
  }

  template <typename Check, template <typename> class Wrap = type_tag,
            typename Visitor = void>
  inline static void visit_lambda_if(Visitor&& visitor, Check check = {}) {
    (void)dummy_list{(
        types_visit_wrap_impl<Wrap, Visitor>::
            template runner<Ts, Check::template check<Ts>::value>::
                run(std::forward<Visitor>(visitor)),
        true)...};
  }
};


template <typename T>
struct type_pack_extract_impl {
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

// Unused.
template <typename T, template <typename...> class Tpl>
using type_pack_extract_constrained =
    typename type_pack_extract_impl<T>::template type_constrained<Tpl>;


// Unused.
template <typename Visitor>
struct types_visit_impl {
  template <typename T, bool execute>
  struct runner {
    inline static void run(Visitor&& visitor) {
      visitor.template run<T>();
    }
  };
  template <typename T>
  struct runner<T, false> {
    inline static void run(Visitor&&) {}
  };
};

// TODO(eric.cousineau): See if there is a way to return result? (if useful)
template <typename Check, typename ... Ts, typename Visitor>
inline void types_visit_if(Visitor&& visitor) {
  // Minor goal: Avoid needing index sequences (reduce number of types?).
  (void)dummy_list{(
      types_visit_impl<Visitor>::
          template runner<Ts, Check::template check<Ts>::value>::
              run(std::forward<Visitor>(visitor)),
      true)...};
}

template <typename ... Ts, typename Visitor>
inline void types_visit(Visitor&& visitor) {
  types_visit_if<always_true, Ts...>(std::forward<Visitor>(visitor));
}
