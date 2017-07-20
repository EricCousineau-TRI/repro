#include <iostream>
#include <string>

using std::string;
using std::cout;
using std::endl;

#define EVAL(x) std::cout << ">>> " #x ";" << std::endl; x; cout << std::endl
#define PRINT(x) ">>> " #x << std::endl << (x) << std::endl

namespace obj {

struct Item {
    Item(int x)
    {
        cout << "ctor" << endl;
    }
};

} // namespace obj

namespace helper {

auto Item(const string& y) {
    cout << "helper" << endl;
    return obj::Item(2);
}

} // namespace helper

using namespace obj;
using namespace helper;

int main() {
    int x {2};
    double y {3.0};
    EVAL(Item(x));
    EVAL(Item());

    return 0;
}
