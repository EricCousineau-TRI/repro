#pragma once

#include <string>

class Test {
public:
    /*
    // Does not work
    template<typename T>
    decltype(auto) tpl_method(const T& x);

    template<>
    std::string tpl_method<int>(const int& x) {
        return "int";
    }

    template<>
    auto tpl_method<double>(const double& x) {
        return 2 * x;
    }
    */

    // Does work
    template<typename T>
    auto tpl_method(const T& x);
};

template<>
auto Test::tpl_method<int>(const int& x) {
    return std::string("int -> string");
}

template<>
auto Test::tpl_method<double>(const double& x) {
    return 2 * x;
}
