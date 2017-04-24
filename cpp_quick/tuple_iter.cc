#include <string>
#include <iostream>
#include <tuple>

#include "tuple_future.h"
#include "name_trait.h"

using std::cout;
using std::endl;

template<typename Callable>
void visit(Callable&& f) { }
template<typename Callable, typename T1, typename... Args>
void visit(Callable&& f, T1&& t1, Args&&... args) {
    std::forward<Callable>(f)(std::forward<T1>(t1));
    visit(std::forward<Callable>(f), std::forward<Args>(args)...);
}
// template<typename Callable, 

int main() {
    // auto t = make_tuple(1, "hello");
    auto f = [=](auto&& x) {
        cout << "visit: " << x << endl;
    };

    visit(f, 1, "hello");

    return 0;
}
