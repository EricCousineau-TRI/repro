#include <iostream>
#include <string>

#include "name_trait.h"

using namespace std;

// using std::cout;
// using std::endl;
// using std::string;

void my_func(string& s) {
    cout << "lvalue reference" << endl;
}

void my_func(string&& s) {
    cout << "rvalue reference" << endl;
}

void my_func(const string& s) {
    cout << "const lvalue reference" << endl;
}

void my_func(const string&& s) {
    cout << "const rvalue reference" << endl;
}

int main() {
    string x = "string";
    const string cx = "const string";
    EVAL(my_func(x));
    EVAL(my_func(static_cast<const string>(string("wut"))));
    EVAL(my_func("hello"));
    EVAL(my_func(cx));
    return 0;
}
