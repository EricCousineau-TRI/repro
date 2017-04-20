// Goal: Ens

#include <iostream>
#include <string>

#include "name_trait.h"

using std::string;
using std::cout;
using std::endl;

template<typename ... Args>
struct pack { };
NAME_TRAIT_TPL(pack);

template<typename ... Args>
void my_func(Args&& ... args) {
    using my_pack = pack<Args&&...>;
    cout << "1. my_func variadic: " << name_trait<my_pack>::name() << endl;
}

// void my_func(const int& x, const string& y) {
//     cout << "2. my_func(const int&, const string&)" << endl;
// }

void my_func(int y, string&& z) {
    cout << "3. my_func(int, string&&)" << endl;
}

int main() {
    EVAL(( my_func(1, 2, string("!!!")) ));
    EVAL(( my_func(2, string("Hello")) ));
    EVAL(( my_func(3, "World") ));
    
    return 0;
}
