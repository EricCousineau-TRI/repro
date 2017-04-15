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

void tpl_func(const int& x);
void tpl_func(const double& x);
