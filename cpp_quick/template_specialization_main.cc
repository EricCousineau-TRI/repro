/*
Goal: Test template specialization with the following goals:

* Determine maximum flexibility on template specialization return types
* Determine resolution based on inheritance
    * Using parameter type vs. argument (e.g. pointer)
* Determine the constraints on explicit instantiations with template specializations, esp. when some of the implementation can be hidden
*/

#include <iostream>

#include "name_trait.h"
#include "template_specialization.h"

using std::cout;
using std::endl;

int main() {
    Test t;

    cout
        << PRINT(t.tpl_method(2))
        << PRINT(t.tpl_method(3.));

    return 0;
}
