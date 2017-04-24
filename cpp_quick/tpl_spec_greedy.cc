// Goal: Use template specialization to greedy match against a
// parameter pack base definition

#include <iostream>
#include <string>
#include <type_traits>

#include "name_trait.h"

using std::string;
using std::cout;
using std::endl;

using std::enable_if;
using std::is_convertible;

template<bool Expr, typename Result = void>
using enable_if_t = typename enable_if<Expr, Result>::type;
template<typename From, typename To>
using enable_if_convertible_t = enable_if_t<is_convertible<From, To>::value, To>;

template<typename ... Args>
void my_func(Args&& ... args) {
    cout << "1. my_func<Args...>(" << name_trait_list<Args&&...>::join() << ")" << endl;
}

// Use template with enable_if to catch as many types as possible
template<typename T1,
    typename Cond = enable_if_convertible_t<T1, string>>
void my_func(int y, T1&& z) {
    cout
        << "2. my_func<T1:string>(int, " << name_trait<decltype(z)>::name()
        << ")" << endl;
}

// Use template with enable_if to catch as many types as possible
template<typename T1,
    typename Cond = enable_if_convertible_t<T1, double>>
void my_func(int y, T1&& z, Cond* = nullptr) { // Have to do weird stuff to catch different templates...
    cout
        << "3. my_func<T1:double>(int, " << name_trait<decltype(z)>::name()
        << ")" << endl;
}

// Example using multiple types (let compiler handle the combinatorics)
template<typename T1, typename T2,
    typename = enable_if_t<
        is_convertible<T1, string>::value && is_convertible<T2, double>::value>>
void my_func(int y, T1&& z, T2&& zz) {
    cout
        << "4. my_func<T1:string, T2:double>(int, "
        << name_trait<decltype(z)>::name() << ", "
        << name_trait<decltype(zz)>::name() << ")" << endl;
}

// // // This alone does not catch string literal
// void my_func(int y, string&& z) {
//     cout << "my_func(int, string&&)" << endl;
// }

// // Cannot use condition as argument type
// template<typename T>
// void my_func(int y, enable_if_convertible_t<string, T>&& z) {
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
