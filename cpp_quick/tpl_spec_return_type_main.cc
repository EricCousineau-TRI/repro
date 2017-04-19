/*
Goal: Test template specialization with the following goals:

* Determine maximum flexibility on template specialization return types
* Determine resolution based on inheritance
    * Using parameter type vs. argument (e.g. pointer)
* Determine the constraints on explicit instantiations with template specializations, esp. when some of the implementation can be hidden
*/

#include <iostream>

#include "name_trait.h"
#include "tpl_spec_return_type.h"

using std::cout;
using std::endl;

int main() {
    int x {2};
    double y {2.};

    Test t;

    cout
        << PRINT(t.tpl_method_auto(x))
        << PRINT(t.tpl_method_auto(y))
        << PRINT(t.tpl_method_explicit(x))
        << PRINT(t.tpl_method_explicit(y));
    cout << "---" << endl;
    cout
        << PRINT(create_value(x))
        << PRINT(create_value<int>(x))
        << PRINT(create_value<int>(1))
        << PRINT((create_value<int, int>(1)));

    return 0;
}
