#include <iostream>
#include <cmath>
using std::cout;
using std::endl;

#include <Eigen/Dense>
// #include <unsupported/Eigen/AutoDiff>
#include "AutoDiffScalarMod.h"

// https://eigen.tuxfamily.org/dox/TopicInsideEigenExample.html
// - Explaining: internal::traits

template <typename DerScalar, int num_vars>
using AutoDiffdBase = Eigen::AutoDiffScalar<double, Eigen::Matrix<DerScalar, num_vars, 1> >;

/* <snippet from="drake-distro:5729940:drake/common/eigen_autodiff_types.h"> */
// // An autodiff variable with `num_vars` partials.
template <int num_vars>
using AutoDiffd = AutoDiffdBase<double, num_vars>;
/* </snippet> */

template<int num_vars>
using AutoDiff2d = AutoDiffdBase<AutoDiffd<num_vars>, num_vars>;

#define PRINT(x) #x ": " << (x) << endl

int main() {
    AutoDiff2d<1> x;
    x.value() = 4;
//    x.derivatives()(0).value() = 1;
//    x.derivatives()(0).derivatives() = 1;

//    auto expr = x * x;
//    cout
//        << PRINT(expr.value())
//        << PRINT(expr.derivatives().value())
//        << PRINT(expr.derivatives().derivatives());

    return 0;
}
