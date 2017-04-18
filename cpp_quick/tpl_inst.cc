#include "tpl_inst.h"

//// Simple method
template<typename T>
void tpl_func(const T& x) {
    std::cout << "template: " << name_trait<T>::name() << std::endl;
}

template void tpl_func<int>(const int& x);
template void tpl_func<double>(const double& x);


//// Variadic
template<typename ... Args>
void tpl_func_var(Args ... args) {
    std::cout << "variadic" << std::endl;
}
template void tpl_func_var(int x, double y);


//// Method - only defined in source
template<typename T>
void test::tpl_method_source(const T& x) {
    std::cout << "method template: " << name_trait<T>::name() << std::endl;
}
// Explicitly instantiate
template void test::tpl_method_source<int>(const int& x);


//// Method - defind in header, but with an explicit instantiation in source
// Explicitly instantiate
template void test::tpl_method_source_spec<int>(const int& x);
