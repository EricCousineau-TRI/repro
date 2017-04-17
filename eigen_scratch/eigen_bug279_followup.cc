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

template<int order>
struct node {
    node<order - 1> next;
    double value;
    node(double value = 0)
        : next(value + 2), value(value)
    { }
};
template<>
struct node<0> {
    double value;
    node(double value = 0)
        : value(value)
    { }
};

// Consider submitting as example for:
// * http://eigen.tuxfamily.org/bz/show_bug.cgi?id=634

// Distraction:
// For nth order derivative, per Alexander Werner's post:
// https://forum.kde.org/viewtopic.php?f=74&t=110376#p268948
// and per Hongkai's TODO
template <int order, int num_vars>
struct AutoDiffNdScalar {
    static_assert(order > 0 && order < 5, "Must have order between 1 and 4");

    typedef typename AutoDiffNdScalar<order - 1, num_vars>::type prev_type;
    typedef Eigen::AutoDiffScalar<Eigen::Matrix<prev_type, num_vars, 1> > type;
};

// Base Case
template<int num_vars>
struct AutoDiffNdScalar<0, num_vars> {
    using type = double;
};
// Alternative: AutoDiffNdScalar<1, num_vars> = AutoDiffd<num_vars>

template <int order, int num_vars>
using AutoDiffNd = typename AutoDiffNdScalar<order, num_vars>::type;

template <int order, int num_vars, int rows>
using AutoDiffNVecd = Eigen::Matrix<AutoDiffNd<order, num_vars>, rows, 1>;

template <int order>
using AutoDiffNVecXd = AutoDiffNVecd<order, Eigen::Dynamic, Eigen::Dynamic>;

#define PRINT(x) #x ": " << (x) << endl

int main() {
    auto pi = numeric_const<double>::pi;
    AutoDiffNVecXd<2> x_taylor(1);
    /*
    Layout for AutoDiffNd<2>

            0   1
        0 [ x   dx  ]
        1 [ dx  ddx ]

        0 - value(), 1 - derivative()

    Investigate minimizing redundancy via:
    * https://en.wikipedia.org/wiki/Automatic_differentiation#High_order_and_many_variables

    */
    auto& x = x_taylor(0).value().value();
    // First order
    auto& deriv = x_taylor(0).derivatives();
    deriv.resize(1);
    auto& xdot = deriv(0).value();
    // Symmetric derivative
    auto& deriv_sym = x_taylor(0).value().derivatives();
    deriv_sym.resize(1);
    auto& xdot_sym = deriv_sym(0);
    // Second order
    auto& dderiv = deriv(0).derivatives();
    dderiv.resize(1);
    auto& xddot = dderiv(0);

    x = 2;
    xdot = 5;
    xdot_sym = xdot;
    xddot = 1;

    // pow(x_taylor(0), 2) - not happy with nested?
    auto expr = x_taylor(0) * x_taylor(0); //sin(x_taylor(0));

    cout
        << PRINT(expr.value().value())
        << PRINT(expr.value().derivatives())
        << PRINT(expr.derivatives()(0).value())
        << PRINT(expr.derivatives()(0).derivatives());

    node<5> tree;
    cout << tree.next.next.next.next.next.value << endl;

    // AutoDiffNVecXd<0> x_bad(1); // Fails as expected

    return 0;
}
