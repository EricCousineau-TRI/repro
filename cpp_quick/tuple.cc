/*
std::make_tuple
std::index_sequence
*/

// @ref http://stackoverflow.com/questions/25885893/how-to-create-a-variadic-generic-lambda
    // auto variadic_generic_lambda = [] (auto&&... param) {};

#include <sstream>
#include <iostream>
#include <utility>
#include <tuple>

#include "name_trait.h"

using std::cout;
using std::endl;

// Specialize name_trait for std::size_t, and use different
template<std::size_t T>
struct literal_trait {
  static std::string name() {
    std::ostringstream os;
    os << T;
    return os.str();
  }
};
template<std::size_t T, std::size_t ... Args>
struct literal_trait_list {
  static std::string join(const std::string& delim = ", ") {
    return literal_trait<T>::name() + delim
        + literal_trait_list<Args...>::join(delim);
  }
};
template<std::size_t T>
struct literal_trait_list<T> {
  static std::string join(const std::string& delim = ", ") {
    return literal_trait<T>::name();
  }
};

#define NAME_TRAIT_TPL_LITERAL(TYPE) \
  template<std::size_t ... Args> \
  struct name_trait<TYPE<Args...>> { \
    static std::string name() { \
      return #TYPE "<" + \
        literal_trait_list<Args...>::join() + ">"; \
      } \
  };
NAME_TRAIT_TPL_LITERAL(std::index_sequence);

// http://stackoverflow.com/questions/25885893/how-to-create-a-variadic-generic-lambda
// http://stackoverflow.com/questions/15904288/how-to-reverse-the-order-of-arguments-of-a-variadic-template-function/15908420#15908420

// http://stackoverflow.com/a/31044718/7829525


// From potential implementations for C++17
namespace future {
/* <snippet from="http://en.cppreference.com/w/cpp/utility/apply"> */
namespace detail {
template <class F, class Tuple, std::size_t... I>
constexpr decltype(auto) apply_impl(F &&f,
    Tuple &&t, std::index_sequence<I...>) 
{
    return f(std::get<I>(std::forward<Tuple>(t))...);
}

}  // namespace detail

template <class F, class Tuple>
constexpr decltype(auto) apply(F &&f, Tuple &&t) 
{
    return detail::apply_impl(
        std::forward<F>(f), std::forward<Tuple>(t),
        std::make_index_sequence<
            std::tuple_size<std::decay_t<Tuple>>::value>{});
}

/* </snippet> */
}

//// Alternative 1: Use folded expressions
template <class F, class Tuple, std::size_t... I>
constexpr decltype(auto) apply_reversed_impl(F &&f,
    Tuple &&t, std::index_sequence<I...>) 
{
    // Reversed
    constexpr std::size_t back_index = sizeof...(I) - 1;
    return f(std::get<back_index - I>(std::forward<Tuple>(t))...);
}

template <class F, class Tuple>
constexpr decltype(auto) apply_reversed(F &&f, Tuple &&t) 
{
    return apply_reversed_impl(
        std::forward<F>(f), std::forward<Tuple>(t),
        std::make_index_sequence<
            std::tuple_size<std::decay_t<Tuple>>::value>{});
}


//// Alternative 2: Use reversed sequences
// http://stackoverflow.com/a/31044718/7829525
template<unsigned N, unsigned... I>
struct reversed_index_sequence
    : reversed_index_sequence<N - 1, I..., N - 1>
{};
template<unsigned... I>
struct reversed_index_sequence<0, I...>
    : std::index_sequence<I...> {
    using sequence = std::index_sequence<I...>;
};
/*
Example:
  rev<3>
    : rev<2,  2>
        : rev<1,  2, 1>
            : rev<0,  2, 1, 0>
                : seq<2, 1, 0>
*/

template<std::size_t N>
struct reversed_index_sequence_trait {
    template <std::size_t... I>
    static auto get_reversed(std::index_sequence<I...>) {
        return reversed_index_sequence<I...>{};
    }
    using type = decltype(get_reversed(std::make_index_sequence<N>{}));
};

template<std::size_t N>
using make_reversed_index_sequence 
    = typename reversed_index_sequence_trait<N>::type;

template <class F, class Tuple>
constexpr decltype(auto) apply_reversed_alt(F &&f, Tuple &&t) 
{
    return future::detail::apply_impl(
        std::forward<F>(f), std::forward<Tuple>(t),
        make_reversed_index_sequence<
            std::tuple_size<std::decay_t<Tuple>>::value>{});
}



//// Alternative 3: Reverse tuple, then use future::apply
// @ref http://stackoverflow.com/questions/25119048/reversing-a-c-tuple
// TODO(eric.cousineau): Try this out... Maybe


double func(int x, double y) {
    cout << "func(int, double)" << endl;
    return x + y;
}
double func(double x, double y) {
    cout << "[reversed] func(double, int)" << endl;
    return x - y;
}

int main() {
    // Make function callable
    // @ref http://nvwa.cvs.sourceforge.net/viewvc/nvwa/nvwa/functional.h?view=markup - Line 453 (lift_optional)
    auto func_callable = [=] (auto&&... args) {
        return func(std::forward<decltype(args)>(args)...);
    };
    cout << func_callable(1, 2.0) << endl;

    auto t = std::make_tuple(1, 2.0);
    future::apply(func_callable, t);
    apply_reversed(func_callable, t);
    // apply_reversed_alt(func_callable, t);

    cout
        << "Sequence: " << endl
        << PRINT(name_trait<decltype(std::make_index_sequence<5> {})>::name())
        << PRINT(name_trait<reversed_index_sequence_trait<5>::type::sequence>::name())
        << endl;

    cout << endl;
    return 0;
}
