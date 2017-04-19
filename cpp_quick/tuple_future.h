#include <utility>
#include <tuple>

// Future implementations from C++17

namespace stdfuture {
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
} // namespace stdfuture


// Custom mechanism to reverse arguments to a callable object

namespace stdcustom {

namespace detail {

template <class F, class Tuple, std::size_t... I>
constexpr decltype(auto) apply_reversed_impl(F &&f,
    Tuple &&t, std::index_sequence<I...>) 
{
    // @ref http://stackoverflow.com/a/31044718/7829525
    // Credit: Orient
    constexpr std::size_t back_index = sizeof...(I) - 1;
    return f(std::get<back_index - I>(std::forward<Tuple>(t))...);
}

} // namespace detail

/**
 * Apply a tuple of arguments to a callable object, F.
 * @param F Callable object, could be generated using variadic auto lambda
 * @param t Tuple of arguments, could be generated from std::forward_as_tuple
 * 
 * Simple Example:
 * 
 *    std::function<void(int, double)> my_func = ...;
 *    stdcustom::apply_reversed(my_func, std::make_tuple(1, 2.0));
 *
 * Extended Example:
 *
 *    void my_func(...) { ... }
 *    ...
 *    auto my_func_reversed = [] (auto&&... revargs) {
 *      auto my_func_callable = [] (auto&& ... args) {
 *        my_func(std::forward<decltype(args)>(args)...);
 *      }
 *      return stdcustom::apply_reversed(my_func_callable,
 *          std::forward_as_tuple(revargs...));
 *    }
 *  };
 */
template <class F, class Tuple>
constexpr decltype(auto) apply_reversed(F &&f, Tuple &&t) 
{
    // Pass sequence by value to permit template inference
    // to parse indices as parameter pack
    return detail::apply_reversed_impl(
        std::forward<F>(f), std::forward<Tuple>(t),
        std::make_index_sequence<
            std::tuple_size<std::decay_t<Tuple>>::value>{});
}

} // namespace stdcustom
