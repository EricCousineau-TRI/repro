// @ref http://stackoverflow.com/questions/32282705/a-failure-to-instantiate-function-templates-due-to-universal-forward-reference

#include <utility>
#include <iostream>

using std::cout;
using std::endl;

template<typename T>
struct is_nested : std::false_type { };

template<template <typename> class T, typename A>
struct is_nested<T<A>> : std::true_type {
    template<typename B>
    using outer_type = T<B>;
    using inner_type = A;
};

// Helper 1: Connect to enable_if
template<typename T>
using enable_if_nested = std::enable_if<
    is_nested<T>::value,
    is_nested<T> // Return info so that we can extract it
    >;

// Helper 2: Connect to std::decay to remove const-ref stuff (will do other simplifications too)
template<typename T>
using enabel_if_nested_decay = enable_if_nested<
        typename std::decay<T>::type
        >;

// Put it into use
template <typename T_A, typename Result =
    typename enabel_if_nested_decay<T_A>::type>
decltype(auto) f (T_A&& t)
{
    // Blech, but necessary
    using A = typename Result::inner_type;
    // // Cannot use template aliases directly :(
    // // @ref http://stackoverflow.com/q/34419603/170413
    // using T = typename Info::outer_type;

    cout << "t.bar = " << t.bar << endl;
    cout << " - default inner: " << A() << endl;

    // Reinstantiate the class with inner type double
    using T_double = typename Result::template outer_type<double>;
    T_double c { .bar = 0.5 * t.bar };
    cout << "T_double: " << c.bar << endl;
    // Permit da forwarding
    return std::forward<T_A>(t);
}

template <typename A>
struct foo
{
    A bar;
};

int main() {
    struct foo<int>        x1 { .bar = 1 };
    struct foo<int> const  x2 { .bar = 2 };
    struct foo<int> &      x3 = x1;
    struct foo<int> const& x4 = x2;

    auto r1 = f (x1);
    auto r2 = f (x2);
    auto r3 = f (x3);
    auto r4 = f (x4);
    auto r5 = f (foo<double> {3.5}); // only rvalue works

    return 0;
}
