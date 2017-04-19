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



double func(int x, double y) {
    cout << "func(int, double)" << endl;
    return x + y;
}

int main() {
    // Make function callable
    // @ref http://nvwa.cvs.sourceforge.net/viewvc/nvwa/nvwa/functional.h?view=markup - Line 453 (lift_optional)
    auto func_callable = [=] (auto&&... args) {
        return func(std::forward<decltype(args)>(args)...);
    };
    cout << func_callable(1, 2.0) << endl;

    auto t = std::make_tuple(1, 2.0);
    apply(func_callable, t);

    cout << endl;
    return 0;
}
