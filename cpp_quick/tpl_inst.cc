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

namespace {
    void instantiate() {
        // using std::declval;
        // decltype(tpl_func(declval<int>())) *a1;
        // decltype(tpl_func(declval<double>())) *a2;
        tpl_func(int{});
        tpl_func(double{});
    }
}

// template<>
// void tpl_func<int>(const int& x);
// template<>
// void tpl_func<double>(const double& x);
