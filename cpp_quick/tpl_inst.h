#pragma once

#include <iostream>

#include "name_trait.h"

// Try to replicate: http://en.cppreference.com/w/cpp/language/class_template#Explicit_instantiation
// Can do so with:   http://en.cppreference.com/w/cpp/language/function_template#Explicit_instantiation

// Providing definition here prevents other implementations of tpl_traits from being used
// Example: move this definiton here, and observe that the traits are still generic
// Does this mean it isn't using the specializations from `tpl_inst.cc`? How do I inspect this?

//// Function
template<typename T>
void tpl_func(const T& x);


//// Variadic Function
template<typename ... Args>
void tpl_func_var(Args ... args);

//// Method
struct test {
    //// - Only defind in source
    // Explicitly rely on external definition (in source file)
    template<typename T>
    void tpl_method_source(const T& x);

    //// - Defined in header, with implicit, explicit, explicit extern, and specialized extern definitions
    template<typename T>
    void tpl_method_source_spec(const T& x) {
        std::cout << "header impl: generic <" << name_trait<T>::name() << ">" << std::endl;
    }
};

// - Do not instantiate here; explicitly instantiate in source file
// Will use header definition
extern template
void test::tpl_method_source_spec<int>(const int& x);

// - Forward declare specialization
template<>
void test::tpl_method_source_spec<bool>(const bool& x);

// // - DOES NOT WORK: Forward declare using extern template
// // Will require explicit instantiation of generic, not a specialization
// extern template
// void test::tpl_method_source_spec<bool>(const bool& x);

// - Specialize in header
// Must use inlin in header, otherwise causes multiple definition errors.
// @ref http://stackoverflow.com/q/1723537/7829525
template<>
inline void test::tpl_method_source_spec<double>(const double& x) {
    std::cout << "header impl: double" << std::endl;
}
