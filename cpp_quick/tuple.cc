/*
std::make_tuple
std::index_sequence
*/

// @ref http://stackoverflow.com/questions/25885893/how-to-create-a-variadic-generic-lambda
    // auto variadic_generic_lambda = [] (auto&&... param) {};

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

template <class F, class Tuple, std::size_t... I>
constexpr decltype(auto) apply_reversed_impl(F &&f,
    Tuple &&t, std::index_sequence<I...>) 
{
    // Reversed
    constexpr std::size_t back_index = sizeof...(I);
    return f(std::get<back_index - I>(std::forward<Tuple>(t))...);
}

}  // namespace detail

// TODO(eric.cousineau): Figure out how 
 
template <class F, class Tuple>
constexpr decltype(auto) apply(F &&f, Tuple &&t) 
{
    return detail::apply_impl(
        std::forward<F>(f), std::forward<Tuple>(t),
        std::make_index_sequence<
            std::tuple_size<std::decay_t<Tuple>>::value>{});
}

template <class F, class Tuple>
constexpr decltype(auto) apply_reversed(F &&f, Tuple &&t) 
{
    return detail::apply_reversed_impl(
        std::forward<F>(f), std::forward<Tuple>(t),
        std::make_index_sequence<
            std::tuple_size<std::decay_t<Tuple>>::value>{});
}

/* </snippet> */
}

/*
// http://stackoverflow.com/a/31044718/7829525
template<unsigned N, unsigned... Indices>
struct reversed_index_sequence
    : reversed_index_sequence<N - 1, Indices..., N - 1>
{};
template<unsigned... Indices>
struct reversed_index_sequence<0, Indices...>
    : std::index_sequence<Indices...>{};

auto get_reversed
template<std::size_t N, unsigned... Indices>
template reversed {
    auto get() {
    }
};

template<std::size_t N>
using make_reversed_index_sequence = reversed_index_sequence<N>;

template <class F, class Tuple>
constexpr decltype(auto) apply_reversed(F &&f, Tuple &&t) 
{
    return detail::apply_impl(
        std::forward<F>(f), std::forward<Tuple>(t),
        make_reversed_index_sequence<
            std::tuple_size<std::decay_t<Tuple>>::value>{});
}
*/


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
    future::apply_reversed(func_callable, t);

    cout << endl;
    return 0;
}
