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

// Distraction:
// For nth order derivative, per Alexander Werner's post:
// https://forum.kde.org/viewtopic.php?f=74&t=110376#p268948
// and per Hongkai's TODO
template <int order>
struct AutoDiffNdTraits {
    static_assert(order > 0 && order < 5, "Must have order between 1 and 4");
    // Previous order
    typedef AutoDiffNdTraits<order - 1> Prev;
    template <int num_vars>
    using PrevScalar = typename Prev::template Scalar<num_vars>;

    template <int num_vars>
    using Scalar = Eigen::AutoDiffScalar<Eigen::Matrix<PrevScalar<num_vars>, num_vars, 1> >;

    // Not necessary to place these here, but keeping for simplicity
    template <int num_vars, int rows>
    using Vector = Eigen::Matrix<Scalar<num_vars>, rows, 1>;

    typedef Vector<Eigen::Dynamic, Eigen::Dynamic> VectorXd;
};

template<>
struct AutoDiffNdTraits<1> {
    template <int num_vars>
    using Scalar = AutoDiffd<num_vars>;
    template <int num_vars, int rows>
    using Vector = AutoDiffVecd<num_vars, rows>;
    using VectorXd = AutoDiffVecXd;
};
/* Alternative:
template<>
struct AutoDiffNdTraits<0> {
    template <int nm_vars>
    using Scalar = double;
};
*/

template <int order, int num_vars>
using AutoDiffNd = typename AutoDiffNdTraits<order>::template Scalar<num_vars>;

template <int order, int num_vars, int rows>
using AutoDiffNVecd = typename AutoDiffNdTraits<order>::template Vector<num_vars, rows>;

template <int order>
using AutoDiffNVecXd = typename AutoDiffNdTraits<order>::VectorXd;

// template <int order, int num_vars, int rows>
// using AutoDiffNVecd = Eigen::Matrix<AutoDiffNd<order, num_vars>, rows, 1>;

// template <int order>
// typedef AutoDiffNVecd<order, Eigen::Dynamic, Eigen::Dynamic> AutoDiffNVecXd;


#define PRINT(x) #x ": " << (x) << endl

int main() {
    auto pi = numeric_const<double>::pi;
    AutoDiffNVecXd<2> x_taylor(1);
    /* Storage:
        [ x   dx  ]
        [ dx  ddx ]
    */
    auto& x = x_taylor(0).value().value();
    // First order
    auto& deriv = x_taylor(0).derivatives();
    deriv.resize(1);
    auto& xdot = deriv(0).value();
    // Symmetric derivative (maintain tree structure...)
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
