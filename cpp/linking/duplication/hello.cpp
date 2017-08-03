#include <iostream>
#include "singleton.h"

extern "C" void hello() {
      singleton::instance().num = 100; // call singleton
    std::cout << "singleton.num in hello.so : " << singleton::instance().num << std::endl;
    ++singleton::instance().num;
    std::cout << "singleton.num in hello.so after ++ : " << singleton::instance().num << std::endl;
    std::cout << singleton::pInstance << std::endl;
    Produce<double>();
}
