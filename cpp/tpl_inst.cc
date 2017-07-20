#include "tpl_inst.h"

//// Function
template<typename T>
void tpl_func(const T& x) {
    std::cout << "template: " << name_trait<T>::name() << std::endl;
}

template void tpl_func<int>(const int& x);
template void tpl_func<double>(const double& x);


//// Variadic Function
template<typename ... Args>
void tpl_func_var(Args ... args) {
    std::cout << "variadic source impl: " << name_trait_list<Args...>::join()
        << std::endl;
}
template void tpl_func_var(int x, double y);


//// Method - only defined in source
template<typename T>
void test::tpl_method_source(const T& x) {
    std::cout << "source impl: " << name_trait<T>::name() << std::endl;
}
// Explicitly instantiate
template void test::tpl_method_source<int>(const int& x);


//// Method - defind in header, but with an explicit instantiation in source
// Explicitly instantiate
template void test::tpl_method_source_spec<int>(const int& x);

// COMPILES, IS NOT SEEN: Explicitly specialize in source
template<>
void test::tpl_method_source_spec<bool>(const bool& x) {
    std::cout << "source specialization impl: bool" << std::endl;
}
