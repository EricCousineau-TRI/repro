#include <iostream>
#include "singleton.h"

extern "C" void hello2() {
    std::cout << "singleton.num in hello2.so : " << singleton::instance().num << std::endl;
    ++singleton::instance().num;
    std::cout << "singleton.num in hello2.so after ++ : " << singleton::instance().num << std::endl;
    std::cout << singleton::pInstance << std::endl;
    Produce<double>();
}
