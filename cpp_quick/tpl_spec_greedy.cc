// Goal: Use template specialization to greedy match against a
// parameter pack base definition

#include <iostream>
#include <string>

#include "name_trait.h"

using std::string;
using std::cout;
using std::endl;

// Base case
template<typename T, typename Arg>
struct is_compat : public std::false_type { };

// Idempotent case is true
template<typename T>
struct is_compat<T, T> : public std::true_type { };

// Simple alias
template<typename T>
using compat_decay_t = std::remove_cv_t<std::decay_t<T>>;

// enable_if specific to compatibility
template<typename T, typename Arg, typename Result = void>
struct enable_if_compat : std::enable_if<
    is_compat<T, compat_decay_t<Arg>>::value, Result>
{ };

template<typename T, typename Arg>
using enable_if_compat_t = typename enable_if_compat<T, Arg>::type;

// Must explicitly specialize
template<>
struct is_compat<string, char*> : public std::true_type { };
template<>
struct is_compat<string, const char*> : public std::true_type { };

// Or:
#define ADD_COMPAT(T, Arg) \
    template<> struct is_compat<T, Arg> : public std::true_type { };

ADD_COMPAT(double, int);
ADD_COMPAT(double, uint);


template<typename ... Args>
void my_func(Args&& ... args) {
    cout << "my_func<variadic>(" << name_trait_list<Args&&...>::join() << ")" << endl;
}

// Use template with enable_if to catch as many types as possible
template<typename T1,
    typename C1 = enable_if_compat_t<string, T1>>
void my_func(int y, T1&& z) {
    cout << "my_func<cond>(int, " << name_trait<decltype(z)>::name() << ")" << endl;
    cout << "  decay_t = " << name_trait<compat_decay_t<T1>>::name() << endl;
}

// Example using multiple types (let compiler handle the combinatorics)
template<typename T1, typename T2,
    typename C1 = enable_if_compat_t<string, T1>,
    typename C2 = enable_if_compat_t<double, T2>>
void my_func(int y, T1&& z, T2&& zz) {
    cout
        << "my_func<cond_2>(int, "
        << name_trait<decltype(z)>::name() << ", "
        << name_trait<decltype(zz)>::name() << ", " << endl
        << "  decay_t_1 = " << name_trait<compat_decay_t<T1>>::name() << endl
        << "  decay_t_2 = " << name_trait<compat_decay_t<T2>>::name() << endl;
}

// // // This alone does not catch string literal
// void my_func(int y, string&& z) {
//     cout << "my_func(int, string&&)" << endl;
// }

// // Cannot use condition as argument type
// template<typename T>
// void my_func(int y, enable_if_compat_t<string, T>&& z) {
//     cout << "4. my_func(int, " << name_trait<T>::name() << ")" << endl;
// }

int main() {
    char var[] = "howdy";
    EVAL(( my_func(1, 2, 5, string("!!!")) ));
    EVAL(( my_func(6, 12.0) ));
    EVAL(( my_func(3, string("Hello")) ));
    EVAL(( my_func(4, (const string&)string("kinda")) ));
    EVAL(( my_func(5, "World") ));
    EVAL(( my_func(6, var) ));
    EVAL(( my_func(7, var, 12) ));
    EVAL(( my_func(9, var, 12.0) ));
    EVAL(( my_func(9, var, var, 12.5) ));
    
    return 0;
}
