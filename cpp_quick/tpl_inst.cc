#include "tpl_inst.h"

template<typename T>
struct tpl_traits {
    static const char* name() {
        return "generic trait";
    }
};

template<typename T>
void tpl_func(const T& x) {
    std::cout << "template: " << tpl_traits<T>::name() << std::endl;
}

template<>
void tpl_func<int>(const int& x);
template<>
void tpl_func<double>(const double& x);
