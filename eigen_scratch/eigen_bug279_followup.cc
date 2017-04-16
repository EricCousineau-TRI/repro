/* <motivation>
Follow-up testing for Bug #279, PR #194 (git sha: 77bb966):
* http://eigen.tuxfamily.org/bz/show_bug.cgi?id=279
* https://bitbucket.org/eigen/eigen/pull-requests/194/relax-mixing-type-constraints-for-binary

Regarding discussion on mailing list:
* http://eigen.tuxfamily.narkive.com/AQoPj2si/using-matrix-autodiff-in-conjunction-with-matrix-double

Purpose:
While it appears that this disucssion is resolved, there are still explicit casts, such as:
* drake-distro:5729940:drake/solvers/constraint.cc:16

   </motivation> */

#include <iostream>
#include <string>
#include <cmath>
using std::cout;
using std::endl;
using std::string;

template<typename T>
struct numeric_const {
    static constexpr T pi = static_cast<T>(M_PI);
};

/* <snippet from="drake-distro:5729940:drake/common/eigen_autodiff_types.h"> */
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

// For nth order derivative, per Alexander Werner's post:
// https://forum.kde.org/viewtopic.php?f=74&t=110376#p268948
// and per Hongkai's TODO
template <int order, int num_vars>
using AutoDiffNd = Eigen::AutoDiffScalar<Eigen::Matrix<AutoDiffNd<order - 1, num_vars>>>;
// Static assert for order <= 0?
template <int num_vars>
using AutoDiffNd<1, num_vars> = AutoDiffd;

template <int order, int num_vars, int rows>
using AutoDiffNVecd = Eigen::Matrix<AutoDiffNd<order, num_vars>, rows, 1>;

template <int order>
typedef AutoDiffNVecd<order, Eigen::Dynamic, Eigen::Dynamic> AutoDiffNVecXd;


#define PRINT(x) #x ": " << (x) << endl

int main() {
    auto pi = numeric_const<double>::pi;
    AutoDiffVecXd x_taylor(1);
    auto& value = x_taylor(0).value();
    auto& deriv = x_taylor(0).derivatives();
    value = pi / 3;
    // Second-order derivatives???
    deriv.resize(2);
    deriv << 10, 100;

    auto expr = sin(x_taylor(0));

    cout << PRINT(expr.value()) << PRINT(expr.derivatives().transpose()) << endl;

    return 0;
}
