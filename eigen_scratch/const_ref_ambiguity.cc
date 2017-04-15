// Purpose: Try to address issue with needing to explicitly overload Eigen::VectorXd AutoDiffVecXd

#include <iostream>
#include <string>
using std::cout;
using std::endl;
using std::string;


/* <snippet from="//drake/common/eigen_autodiff_types.h"> */
#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>

/// An autodiff variable with `num_vars` partials.
template <int num_vars>
using AutoDiffd = Eigen::AutoDiffScalar<Eigen::Matrix<double, num_vars, 1> >;

/// A vector of `rows` autodiff variables, each with `num_vars` partials.
template <int num_vars, int rows>
using AutoDiffVecd = Eigen::Matrix<AutoDiffd<num_vars>, rows, 1>;

/// A dynamic-sized vector of autodiff variables, each with a dynamic-sized
/// vector of partials.
typedef AutoDiffVecd<Eigen::Dynamic, Eigen::Dynamic> AutoDiffVecXd;
/* </snippet> */


auto vec_ref_explicit(const Eigen::Ref<const Eigen::VectorXd> &x) {
    return string("Eigen::VectorXd");
}
auto vec_ref_explicit(const Eigen::Ref<const AutoDiffVecXd> &x) {
    return string("AutoDiffVecXd");
}

#define PRINT(x) #x ": " << (x) << endl

int main() {
    Eigen::VectorXd x;
    AutoDiffVecXd x_taylor;

    cout
        << PRINT(vec_ref_explicit(x))
        << PRINT(vec_ref_explicit(x_taylor));
    return 0;
}
