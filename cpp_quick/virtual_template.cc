#include <iostream>
using std::cout;
using std::endl;

// Most likely can't have virtual template functions, but I want to see the error
// Yup, can't:
/*
cpp_quick/virtual_template.cc:16:5: error: 'virtual' cannot be specified on member function templates
    virtual void tpl();
    ^~~~~~~~
*/

class Base {
public:
    template<typename T>
    virtual void tpl();
};

int main() {
    return 0;
}
