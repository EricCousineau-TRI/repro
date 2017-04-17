#include <iostream>
#include <cmath>
using std::cout;
using std::endl;

#include <Eigen/Dense>
// #include <unsupported/Eigen/AutoDiff>
#include "AutoDiffScalarMod.h"

/* <snippet from="drake-distro:5729940:drake/common/eigen_autodiff_types.h"> */
// // An autodiff variable with `num_vars` partials.
 template <int num_vars>
 using AutoDiffd = Eigen::AutoDiffScalar<double, Eigen::Matrix<double, num_vars, 1> >;


template<int num_vars>
using AutoDiff2d = Eigen::AutoDiffScalar<double, AutoDiffd<num_vars>>;
/* </snippet> */

#define PRINT(x) #x ": " << (x) << endl

int main() {
    AutoDiff2d<1> x;
    x.value() = 4;
    x.derivatives()(0).value() = 1;
    x.derivatives()(0).derivatives() = 1;

    auto expr = x * x;
    cout
        << PRINT(expr.value())
        << PRINT(expr.derivatives().value())
        << PRINT(expr.derivatives().derivatives());

    return 0;
}
