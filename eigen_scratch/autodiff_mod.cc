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

template<typename Scalar>
auto myfun(const Scalar &x) {
    return 2 * x; // * x;
}

//void first_deriv() {
//    AutoDiffd<1> x;
//    x.value() = 4;
//    x.derivatives()(0) = 1;
//
//    auto expr = myfun(x);
//    cout
//        << PRINT(expr.value())
//        << PRINT(expr.derivatives());
//}

void second_deriv() {
    AutoDiff2d<1> x;
    x.value() = 4;
    auto& deriv = x.derivatives()(0);
    deriv.value() = 1.;
    deriv.derivatives()(0) = 1.;

    auto expr = myfun(x);
    cout
        << PRINT(expr.value())
        << PRINT(expr.derivatives()(0).value());
//        << PRINT(expr.derivatives()(0).derivatives());
}

int main() {
//    first_deriv();
    second_deriv();
    return 0;
}
