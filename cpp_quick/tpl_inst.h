#include <iostream>

// Try to replicate: http://en.cppreference.com/w/cpp/language/class_template#Explicit_instantiation
// Can do so with:   http://en.cppreference.com/w/cpp/language/function_template#Explicit_instantiation

template<typename T>
struct tpl_traits {
    static const char* name() {
        return "generic trait";
    }
};

template<typename T>
void tpl_func(const T& x);
