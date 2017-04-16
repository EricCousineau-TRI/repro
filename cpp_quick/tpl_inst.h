#include <iostream>

// Try to replicate: http://en.cppreference.com/w/cpp/language/class_template#Explicit_instantiation
// Can do so with:   http://en.cppreference.com/w/cpp/language/function_template#Explicit_instantiation

template<typename T>
struct tpl_traits {
    static const char* name() {
        return "generic trait";
    }
};

// Providing definition here prevents other implementations of tpl_traits from being used
// Example: move this definiton here, and observe that the traits are still generic
// Does this mean it isn't using the specializations from `tpl_inst.cc`? How do I inspect this?
template<typename T>
void tpl_func(const T& x);

// Specialized methods
struct test {
    template<typename T>
    void tpl_method(const T& x);
};
