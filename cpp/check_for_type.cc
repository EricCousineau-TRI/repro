#include <iostream>
#include <utility>

using std::cout;
using std::endl;

struct trait_good {
    typedef int type;
};
struct trait_bad {
    typedef int wrong_name;
};

template <typename T>
struct defer : std::is_same<T, T> { };

template <typename T, typename C = void>
struct info : std::false_type { };

template <typename T>
struct info<T>
    : std::enable_if<
        defer<typename T::type>::value,
        std::true_type> { };

template <typename T>
void print() {
    cout << "has ::type? " << info<T>::value << endl;
}

int main() {
    print<trait_good>();
    print<trait_bad>();
    return 0;
}
