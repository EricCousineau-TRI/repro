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
constexpr decltype(auto) apply_reversed_alt1_impl(F &&f,
    Tuple &&t, std::index_sequence<I...>) 
{
    // Reversed
    constexpr std::size_t back_index = sizeof...(I) - 1;
    return f(std::get<back_index - I>(std::forward<Tuple>(t))...);
}

template <class F, class Tuple>
constexpr decltype(auto) apply_reversed_alt1(F &&f, Tuple &&t) 
{
    return apply_reversed_alt1_impl(
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
    using sequence = std::index_sequence<I...>; // using type = seq
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
using make_reversed_index_sequence 
    = typename reversed_index_sequence<N>::sequence;

template <class F, class Tuple>
constexpr decltype(auto) apply_reversed_alt2(F &&f, Tuple &&t) 
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

#define CALL(x) cout << ">>> " #x << endl; x; cout << endl;

int main() {
    //// Example sequences
    cout
        << PRINT(name_trait<std::make_index_sequence<5>>::name())
        << PRINT(name_trait<make_reversed_index_sequence<5>>::name())
        << endl;

    // Make function callable by name... ish
    // @ref http://nvwa.cvs.sourceforge.net/viewvc/nvwa/nvwa/functional.h?view=markup - Line 453 (lift_optional)
    auto func_callable = [=] (auto&&... args) {
        return func(std::forward<decltype(args)>(args)...);
    };

    //// Tuple invocations
    auto t = std::make_tuple(1, 2.0);
    cout
        << PRINT((func_callable(1, 2.0)))
        << PRINT((future::apply(func_callable, t)))
        << PRINT((apply_reversed_alt1(func_callable, t)))
        << PRINT((apply_reversed_alt2(func_callable, t)));

    //// Via argument invocations
    auto func_reversed = [=] (auto&&... args) {
        return apply_reversed_alt1(func_callable,
            std::forward_as_tuple(args...));
    };

    cout
        << PRINT(func_reversed(1, 2.0));

    return 0;
}
