#include <iostream>
using std::cout;
using std::endl;

template <int x>
struct Template {
    static constexpr int value = x;
};

template <typename T>
struct extract_param {};
// Extract integer argument from a given template class.
template <template <int> class Template, int x>
struct extract_param<Template<x>> {
    static constexpr int value = x;
};

int main() {
    using Type = Template<3>;
    cout << Type::value << endl;
    cout << extract_param<Type>::value << endl;
}
