
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

template<typename T, typename... Args>
struct is_all_good_unfriendly {
    static constexpr bool value =
        is_good<T>::value ? is_all_good_unfriendly<Args...>::value : false;
};
template<typename T>
struct is_all_good_unfriendly<T> : is_good<T> { };


// Ensure that name_trait_list is unfolded
template<typename... Args>
struct is_good<name_trait_list<Args...>>
    : is_all_good_unfriendly<Args...> { };


template<typename... Args> 
typename std::enable_if<is_all_good_unfriendly<Args...>::value, void>::type
my_func(Args&&... args) {
    cout << name_trait_list<Args...>::join() << endl;
}

int main() {
    // Compilable:
    my_func(int{},
        string{},
        name_trait_list<int,string,double> {},
        2.0);
    // Output:
    // int, std::string, name_trait_list<int, std::string, double>, double

    double array[2] = {2.0, 3.0};
    my_func(array);
    cout
        << is_good<double[2]>::value << endl
        << is_all_good_unfriendly<double[2]>::value << endl;
    return 0;
}
