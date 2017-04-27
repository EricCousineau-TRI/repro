#include <iostream>

using std::cout;
using std::endl;

template <typename T>
struct A {
public:
    A(int value)
        : value(value_) {}
protected:
    int value_;
};

struct B {
    template <typename T>
    friend class A;

    void stuff(const A& x) {
        cout << "value: " << x << endl;
    }
};

int main() {
    A a {2};
    B b;
    b.stuff(a);
    return 0;
}
