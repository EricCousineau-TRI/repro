#include <iostream>

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
