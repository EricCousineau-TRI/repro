// Goal: Empirically determine lifetimes
// Purpose: Too lazy to fully parse:
// @ref http://en.cppreference.com/w/cpp/language/lifetime

#include <iostream>
#include <utility>

#include "name_trait.h"

using std::cout;
using std::endl;

template <int T>
class Lifetime {
public:
    Lifetime() {
        cout << "ctor (default): " << T << endl;
    }
    Lifetime(const Lifetime&) {
        cout << "copy ctor (const lvalue): " << T << endl;
    }
    Lifetime(Lifetime&&) {
        cout << "copy ctor (rvalue): " << T << endl;
    }
    template <int U>
    Lifetime(const Lifetime<U>&) {
        cout
             << "ctor (const Lifetime<" << U << ">&): "
             << T << endl;
    }
    template <int U>
    Lifetime(Lifetime<U>&&) {
        cout
             << "ctor (Lifetime<" << U << ">&&): "
             << T << endl;
    }
    ~Lifetime() {
        cout << "dtor: " << T << endl;
    }
protected:
    using Base = Lifetime<T>;
};

void func_in(const Lifetime<1>&) {
    cout << "func_in(const Lifetime<1>&)" << endl;
}

Lifetime<4> func_out() {
    cout << "func_out()" << endl;
    return Lifetime<4>();
}


int main() {
    EVAL(Lifetime<1> obj1{});
    EVAL({ Lifetime<2>(); });
    EVAL({ Lifetime<3> obj3 = obj1; });
    EVAL({ Lifetime<3> obj3 = Lifetime<1>(); });
    EVAL(func_in(Lifetime<3>()));
    EVAL({ func_out(); cout << "finish" << endl; });
    EVAL({ const Lifetime<4>& obj4 = func_out(); cout << "finish" << endl; });
    EVAL({ Lifetime<4>&& obj4 = func_out(); cout << "finish" << endl; });

    return 0;
}
