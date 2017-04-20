// Goal: Ens

#include <iostream>
#include <string>

#include "name_trait.h"

using std::string;
using std::cout;
using std::endl;

// By default, incompatible
template<typename T, typename Arg>
struct compat { using type = Arg; };

template<typename T, typename Arg>
using compat_t = typename compat<T, Arg>::type;

template<typename T, typename Arg>
using compat_enable_t = typename std::enable_if<
    std::is_same<compat_t<T, Arg>, Arg>::value, Arg>::type;

template<>
struct compat<string, const char[]> { using type = const char[]; };
template<std::size_t N>
struct compat<string, const char[N]> { using type = const char[N]; };
template<>
struct compat<string, const char*> { using type = const char*; };

template<typename ... Args>
void my_func(Args&& ... args) {
    cout << "1. my_func<variadic>(" << name_trait_list<Args&&...>::join() << ")" << endl;
}

// void my_func(const int& x, const string& y) {
//     cout << "2. my_func(const int&, const string&)" << endl;
// }

// Does not catch string literal...
void my_func(int y, string&& z) {
    cout << "3. my_func(int, string&&)" << endl;
}

// // Still does not catch string literal...
// template<typename T>
// void my_func(int y, compat_enable_t<string, T>&& z) {
//     cout << "4. my_func(int, " << name_trait<T>::name() << ")" << endl;
// }

template<typename T, typename Cond = compat_enable_t<string, T>>
void my_func(int y, T&& z) {
    cout << "4. my_func<cond>(int, " << name_trait<T>::name() << ")" << endl;
}

int main() {
    EVAL(( my_func(1, 2, string("!!!")) ));
    EVAL(( my_func(2, string("Hello")) ));
    EVAL(( my_func(3, "World") ));
    
    return 0;
}
