// Goal: Alias to make_shared<C>(...)
#include <memory>
#include <iostream>

#include "name_trait.h"

using std::cout;
using std::endl;
using std::make_shared;


struct Test {
    template<typename... Args>
    Test(Args&&... args)
    {
        cout << "Test(" << name_trait_list<Args...>::join() << ")" << endl;
    }
};

// // template<typename T>
// struct Factory {
//     // template<typename... Args>
//     // static std::shared_ptr<T> make_shared(Args&&... args) {
//     //     return make_shared<T>(std::forward<Args>(args)...);
//     // }
//     static void test() { }
// };

// using CreateTest = Factory::test;

int main() {
    Test(1);
    Test(1, 2);
    make_shared<Test>(1, 2, 3);
    return 0;
}
