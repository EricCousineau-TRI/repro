// @ref http://en.cppreference.com/w/cpp/language/friend#Template_friends

#include <iostream>

using std::cout;
using std::endl;

template<typename T>
struct A {
    void stuff(const T& x) {
        cout << "value: " << x.value_ << endl;
    }
};

struct B {
    template <typename T>
    friend struct A;

    public:
    B(int value)
        : value_(value) {}
protected:
    int value_;
};

int main() {
    B b {2};
    A<B> a;
    a.stuff(b);
    return 0;
}
