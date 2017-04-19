#pragma once

#include <string>
#include <iostream>

#include "name_trait.h"

class Test {
public:
    
    // // [decltype_auto]
    // //   Only works if specializations return 'decltype(auto)'
    // template<typename T>
    // decltype(auto) tpl_method_decltype_auto(const T& x);

    // [auto]
    //   Only works if specializations have 'auto'
    template<typename T>
    auto tpl_method_auto(const T& x);

    template<typename T> struct args {
        using return_type = T;
    };

    template<typename T>
    typename args<T>::return_type tpl_method_explicit(const T& x) {
        return 2 * x;
    }
};

// [auto / decltype_auto, defined_in_header]

// // + [spec_return_specific]
// //     Does not work
// template<>
// std::string Test::tpl_method_decltype_auto<int>(const int& x) {
//     return "int";
// }

// template<>
// double Test::tpl_method_decltype_auto<double>(const double& x) {
//     return 2 * x;
// }

// + [spec_return_auto]
//     Works
template<>
inline auto Test::tpl_method_auto<int>(const int& x) {
    return std::string("int -> string");
}

template<>
inline auto Test::tpl_method_auto<double>(const double& x) {
    return 2 * x;
}

// + [spec_return_explicit]
//     Works
template<>
struct Test::args<int> {
    using return_type = std::string;
};

template<>
std::string Test::tpl_method_explicit<int>(const int& x);

//// Function: Return direct reference (pass through) in certain overload
template<typename C, typename T>
auto create_value(const T& x) {
    std::cout << "[ create: " << name_trait<C>::name()
        << " + " << name_trait<T>::name() << " ]   ";
    C out {};
    return out + x;
}

// - Speciailize to return pass through
// NOTE: This may not be viewed as a specialization, but rather
// a separate template
template<typename C>
C& create_value(C& x) {
    std::cout << "[ pass through: " << name_trait<C>::name() << " ]   ";
    return x;
}
