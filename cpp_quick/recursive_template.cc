#include <iostream>
using std::cout;
using std::endl;

template<int order>
struct node {
    node<order - 1> next;
    double value;
    node(double value = 0)
        : next(value + 2), value(value)
    { }
};
template<>
struct node<0> {
    double value;
    node(double value = 0)
        : value(value)
    { }
};

int main() {
    node<5> tree;
    cout << tree.next.next.next.next.next.value << endl;
    return 0;
}
