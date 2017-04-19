#include <iostream>
#include <string>
#include <memory>

#include "name_trait.h"

using std::string;
using std::shared_ptr;
using std::make_shared;

NAME_TRAIT_TPL(shared_ptr);

template<typename T>
shared_ptr<auto> tpl_func(const T& x) {
    return make_shared<T>(x);
}

template<>
shared_ptr<auto> tpl_func(const int& x) {
    return make_shared<string>("int -> string");
}

int main() {
    int x {2};
    double y {3.};

    cout
        << PRINT(*tpl_func(x))
        << PRINT(*tpl_func(y));

    return 0;
}

/* error:

cpp_quick/auto_inference.cc:14:12: error: 'auto' not allowed in template argument
shared_ptr<auto> tpl_func(const T& x) {
           ^~~~

*/
