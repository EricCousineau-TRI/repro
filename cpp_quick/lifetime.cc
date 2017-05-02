// Goal: Empirically determine lifetimes
// Purpose: Too lazy to fully parse:
// @ref http://en.cppreference.com/w/cpp/language/lifetime


#include <iostream>

#include "name_trait.h"

using std::cout;
using std::endl;

template <typename T>
class Lifetime {
public:
    Lifetime() {
        cout << "ctor (default): " << name_trait<T>::name() << endl;
    }
    Lifetime(const Lifetime&) {
        cout << "copy ctor (const lvalue): " << name_trait<T>::name() << endl;
    }
    template <typename U>
    Lifetime(const Lifetime<U>&) {
        cout
             << "copy ctor (const lvalue, implicit copy): "
             << name_trait<T>::name() << " <- " << name_trait<U>::name()
             << endl;
    }
    Lifetime(Lifetime&&) {
        cout << "copy ctor (rvalue): " << name_trait<T>::name() << endl;
    }
    ~Lifetime() {
        cout << "dtor: " << name_trait<T>::name() << endl;
    }
protected:
    using Base = Lifetime<T>;
};
NAME_TRAIT_TPL(Lifetime);

void func(const Lifetime<int>&) {
    cout << "func(const lvalue)" << endl;
}

int main() {
    EVAL(Lifetime<int> obj{});
    EVAL({ Lifetime<int>(); });

    return 0;
}
