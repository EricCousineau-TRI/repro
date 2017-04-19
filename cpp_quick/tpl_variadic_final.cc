#include <iostream>

#include "name_trait.h"

using std::cout;
using std::endl;

void my_func(int x) {
    cout << "my_func(x)" << endl;
}
void my_func(int x, int y) {
    cout << "my_func(x, y)" << endl;
}

template<typename ... Args>
void tpl_func(Args ... args, int x) {
    cout << "tpl_func" << endl;
    my_func(args...);
}

int main() {
    tpl_func(1, 2, 3);
    tpl_func(1, 2);
}
