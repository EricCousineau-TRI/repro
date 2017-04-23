/*
Purpose: sAme as the other matrix_stack* attempts, but this time, use tuples and explicit calling out of
hstack() and vstack() for concatenation.

*/
#include "cpp_quick/name_trait.h"

#include <string>
#include <iostream>
#include <memory>

#include <Eigen/Core>
#include <Eigen/Dense>

using std::cout;
using std::endl;
using std::string;
using std::unique_ptr;
