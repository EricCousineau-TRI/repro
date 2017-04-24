#include "tuple_iter.h"

#include <string>
#include <iostream>

#include "tuple_future.h"
#include "name_trait.h"

using std::cout;
using std::endl;

int main() {
    int i = 0;
    auto f = [&](auto&& x) {
        cout << "visit[" << i << "]: " << x << endl;
        i += 1;
    };

    visit_args(f, 1, "hello");

    i = 0;
    auto t = std::make_tuple(1, "hello", 10.5);
    visit_tuple(f, t);

    return 0;
}
