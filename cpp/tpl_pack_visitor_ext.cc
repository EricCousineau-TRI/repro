#include <iostream>
#include <typeinfo>
#include <utility>

// - BEGIN: Added
template <int N, int K, typename T, typename ... Ts>
struct type_at_impl {
  using type = typename type_at_impl<N, K + 1, Ts...>::type;
};

template <int N, typename T, typename ... Ts>
struct type_at_impl<N, N, T, Ts...> {
  using type = T;
};

template <int N, typename ... Ts>
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
struct is_different_than {
  template <typename U>
  using check = negation<std::is_same<T, U>>;
};

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
    inline static void run(...) {}
  };
};

template <typename Check, typename ... Ts, typename Visitor>
inline void types_visit_if(Visitor&& visitor) {
  // Minor goal: Avoid needing index sequences (reduce number of types?).
  int dummy[] = {(
      types_visit_impl<Visitor>::
          template runner<Ts, Check::template check<Ts>::value>::
              run(std::forward<Visitor>(visitor)),
      0)...};
  (void)dummy;
}

template <typename ... Ts, typename Visitor>
inline void types_visit(Visitor&& visitor) {
  types_visit_if<always_true, Ts...>(std::forward<Visitor>(visitor));
}

template <typename ... Ts>
struct type_pack {
  template <template <typename...> class Tpl>
  using bind = Tpl<Ts...>;

  template <int N>
  using type = typename type_at<N, Ts...>::type;

  template <typename Visitor>
  inline static void visit(Visitor&& visitor) {
    types_visit<Ts...>(std::forward<Visitor>(visitor));
  }

  template <typename Check, typename Visitor>
  inline static void visit_if(Visitor&& visitor) {
    types_visit_if<Check, Ts...>(std::forward<Visitor>(visitor));
  }
};

using namespace std;

struct visitor_test {
  int arg_1;
  double arg_2;

  template <typename T>
  void run() {
    cout << "T: " << typeid(T).name() << endl;
    cout << arg_1 << " - " << arg_2 << endl;
  }
};

int main() {
  using Pack = type_pack<double, int>;
  Pack::visit(visitor_test{1, 2.0});

  Pack::visit_if<is_different_than<int>>(visitor_test{5, 15.});

  cout << typeid(Pack::type<0>).name() << endl;
  cout << typeid(Pack::type<1>).name() << endl;

  return 0;
}
