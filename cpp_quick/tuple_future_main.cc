#include <iostream>

#include "tuple_future.h"

using std::cout;
using std::endl;

#define PRINT(x) ">>> " #x << std::endl << (x) << std::endl


double my_func(int x, double y) {
    cout << "func(int, double)" << endl;
    return x + y;
}

double my_func(double x, int y) {
    cout << "[ reversed ] func(double, int)" << endl;
    return x - y;
}


auto my_func_callable = [] (auto&& ... args) {
    return my_func(std::forward<decltype(args)>(args)...);
};

void simple_example() {
    auto t = std::make_tuple(1, 2.0);
    cout
        << PRINT(stdfuture::apply(my_func_callable, t))
        << PRINT(stdcustom::apply_reversed(my_func_callable, t));
}

template <typename F>
auto make_callable_reversed(F&& f) {
    return [] (auto&& ... revargs) {
        return stdcustom::apply_reversed(my_func_callable,
            std::forward_as_tuple(revargs...));
     };
}

void advanced_example() {
    auto my_func_reversed = make_callable_reversed(my_func_callable);

    cout
        << PRINT((my_func_reversed(2.0, 1)))
        << PRINT((my_func_reversed(1, 2.0)));
}

int main() {
    simple_example();
    advanced_example();
}
