
#include <iostream>
#include <string>
using std::string;
using std::cout;
using std::endl;

#include "name_trait.h"

template<typename T>
struct is_good : std::true_type { };

// Make it fail only on double[2]
template<>
struct is_good<double[2]> : std::false_type { };

// Unfriendly version: just return false
template<typename T>
struct is_good_decay
    : is_good<typename std::remove_reference<T>::type>
{ };
/*
Error:
cpp_quick/tpl_check_errors.cc:50:5: error: no matching function for call to 'my_func'
    my_func(int{},
    ^~~~~~~
cpp_quick/tpl_check_errors.cc:40:25: note: candidate template ignored: disabled by 'enable_if' [with Args = <int, std::__cxx11::basic_string<char>, name_trait_list<int, std::__cxx11::basic_string<char>, double>, double (&)[2], double>]
typename std::enable_if<is_all_good<Args...>::value, void>::type
                        ^
*/

template<typename T, typename... Args>
struct is_all_good {
    static constexpr bool value =
        is_all_good<T>::value ?
            is_all_good<Args...>::value : false;
};
template<typename T>
struct is_all_good<T>
    : is_good_decay<T> { };

// Ensure that name_trait_list is unfolded
template<typename... Args>
struct is_good<name_trait_list<Args...>>
    : is_all_good<Args...> { };

template<typename... Args> 
typename std::enable_if<is_all_good<Args...>::value, void>::type
my_func(Args&&... args) {
    cout
        << name_trait_list<typename std::remove_reference<Args>::type...>::join()
        << endl;
}

int main() {
    double array[2] = {2.0, 3.0};
    // Compilable:
    my_func(int{},
        string{},
        name_trait_list<int,string,double> {},
        array,
        2.0);
    // Output:
    // int, std::string, name_trait_list<int, std::string, double>, double
    
    // my_func<double[2]>(array);
    cout
        << is_good<double[2]>::value << endl
        << is_all_good<double[2]>::value << endl;
    return 0;
}
