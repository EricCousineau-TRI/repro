// @ref http://stackoverflow.com/questions/32282705/a-failure-to-instantiate-function-templates-due-to-universal-forward-reference

#include <utility>
#include <iostream>

using std::cout;
using std::endl;

// Use pack to store / access parameters
template<typename T, typename... Args>
struct pack {
    using first = T;
    using subpack = pack<Args...>;
    static constexpr std::size_t size = 1 + sizeof...(Args);
};
template<typename T>
struct pack<T> {
    using first = T;
    static constexpr std::size_t size = 1;
};


// Bug Workaround
// Goal is to start with a parameter pack, and avoid having clang interpret an
// empty parameter pack incorrectly
// NOTE: Was unable to place this in the struct definition...
template<template <typename...> class T, typename... BArgs>
struct nested_redecl {
    using type = T<BArgs...>;
};
// using nested_redecl = T<BArgs...>;

// Base Case
template<typename T>
struct is_nested : std::false_type { };

// General Case
template<template <typename...> class T, typename A, typename... Args>
struct is_nested<T<A, Args...>> : std::true_type {
    using pack = pack<A, Args...>;

    using first_inner_type = A;

    template<typename B, typename... BArgs>
    using outer_template = typename nested_redecl<T, B, BArgs...>::type;

    // See Bug Workaround:
    // The following does not work:
    /*
    template<typename B, typename... BArgs>
    using outer_type = T<B, BArgs...>;

    Error:
    cpp_quick/tpl_tpl_forward.cc:32:24: error: too many template arguments for class template 'foo'
        using outer_type = T<B, BArgs...>;
                           ^    ~~~~~~~~~
    
    Ideally, should not happen, as BArgs is empty in this case
    */
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
    using A = typename Result::first_inner_type;
    // // Cannot use template aliases directly :(
    // // @ref http://stackoverflow.com/q/34419603/170413
    // using T = typename Info::outer_template;

    cout << "t.bar = " << t.bar << endl;
    cout << " - default inner: " << A() << endl;
    cout << " - pack size: " << Result::pack::size << endl;

    if (Result::pack::size == 1) {
        // Reinstantiate the class with inner type double
        using T_double = typename Result::template outer_template<double>;
        T_double t_double { .bar = 0.5 * t.bar };
        cout << " - t_double.bar: " << t_double.bar << endl;
    }

    // Permit da forwarding
    return std::forward<T_A>(t);
}

template <typename A>
struct foo
{
    A bar;
};

template<typename A, typename B>
struct baz
{
    A bar;
    B boo;
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
    auto r5 = f (foo<double> {3.5});

    // struct baz<int, double> const&  x6 = { .bar = 1, .boo = 1.5 };
    // auto r6 = f (x6);

    return 0;
}
