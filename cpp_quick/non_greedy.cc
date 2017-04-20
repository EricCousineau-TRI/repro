// Goal: Ens

#include <iostream>
#include <string>

#include "name_trait.h"

using std::string;
using std::cout;
using std::endl;

template<typename T, typename Arg>
struct is_compat : public std::false_type { };

template<typename T, typename Arg>
using compat_enable_t = typename std::enable_if<
    is_compat<T, std::remove_cv_t<std::decay_t<Arg>>>::value, void>::type;

// template<>
// struct compat<string, const char[]> { using type = const char[]; };
template<>
struct is_compat<string, char*> : public std::true_type { };
template<>
struct is_compat<string, const char*> : public std::true_type { };
// template<>
// struct compat<string, const char*> { using type = const char*; };

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
    char var[] = "howdy";
    EVAL(( my_func(1, 2, string("!!!")) ));
    EVAL(( my_func(2, string("Hello")) ));
    EVAL(( my_func(3, "World") ));
    EVAL(( my_func(3, var) ));
    
    return 0;
}
