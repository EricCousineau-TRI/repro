// Goal: Check that a reference can be stored via a template parameter
// Purpose: Dunno yet. Maybe for ../eigen_scratch/matrix_stack.cc?

#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "name_trait.h"

using std::string;
using std::cout;
using std::endl;
using std::vector;
using std::ostream;

// http://stackoverflow.com/questions/23479015/non-type-reference-template-parameters-and-linkage
// Result: Most likely can't operate on the reference...

template<const int& x>
void func() { }

int main() {
    int x = 10;
    func<(const int&)x>();
    cout << x << endl;
    return 0;
}
