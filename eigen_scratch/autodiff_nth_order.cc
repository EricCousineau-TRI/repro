#include <iostream>
#include <cmath>
using std::cout;
using std::endl;

#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>

// Consider submitting as example for:
// * http://eigen.tuxfamily.org/bz/show_bug.cgi?id=634

// Following Hongkai's TODO: drake-distro:f62d147:drake/common/eigen_autodiff_types.h:23
// For nth order derivative, following Alexander Werner's post:
// https://forum.kde.org/viewtopic.php?f=74&t=110376#p268948
template <int order, int num_vars>
struct AutoDiffNdScalar {
    static_assert(order > 0 && order <= 4, "Must have order between 1 and 4");
    typedef typename AutoDiffNdScalar<order - 1, num_vars>::type prev_type;
    typedef Eigen::AutoDiffScalar<Eigen::Matrix<prev_type, num_vars, 1> > type;
};

// Base Case
template<int num_vars>
struct AutoDiffNdScalar<0, num_vars> {
    using type = double;
};
// Alternative: Define AutoDiffNdScalar<1, num_vars>

template <int order, int num_vars>
using AutoDiffNd = typename AutoDiffNdScalar<order, num_vars>::type;

template <int order, int num_vars, int rows>
using AutoDiffNVecd = Eigen::Matrix<AutoDiffNd<order, num_vars>, rows, 1>;

template <int order>
using AutoDiffNVecXd = AutoDiffNVecd<order, Eigen::Dynamic, Eigen::Dynamic>;

/*
Matrix layout for AutoDiffNd<2>

        v   d
    v [ x   dx  ]
    d [ dx  ddx ]

    v - value(), d - derivatives()

Recursive layout

    X -- v -- v=x
         |     |
         | -- d=dx
         |
         d -- v=dx
               |
              d=ddx

Can be generalized for AutoDiffNd<n> as tensor
    See this by extending above matrix to additional dimensions
Example: AutoDiffNd<3> will have 8 elements,
    value: count - indices
    ---
    x:    1 - vvv
    dx:   3 - vvd vdv dvv
    ddx:  3 - vdd dvd ddv
    dddx: 1 - ddd

There will be symmetric duplication, as noted here:
* https://en.wikipedia.org/wiki/Automatic_differentiation#High_order_and_many_variables

You must manually maintain this duplication if you wish to use the existing implementation

TODO:
* Investigate minimizing redundancy via above link
* Review MATLAB implementation:
    * drake-distro:5729940:drake/matlab/util/geval.m
*/

#define PRINT(x) #x ": " << (x) << endl

int main() {

    // AutoDiffNd<0, 1> x_bad(1); // Fails as expected
    AutoDiffNd<2, 1> x_taylor(1);

    auto& x = x_taylor.value().value();
    // First order
    auto& deriv = x_taylor.derivatives();
    deriv.resize(1);
    auto& xdot = deriv(0).value();
    // Symmetric derivative
    auto& deriv_sym = x_taylor.value().derivatives();
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

    // pow(x_taylor, 2) - errors out...
    // sin(x_taylor);
    auto expr = x_taylor * x_taylor;

    cout
        << PRINT(expr.value().value())
        << PRINT(expr.value().derivatives())
        << PRINT(expr.derivatives()(0).value())
        << PRINT(expr.derivatives()(0).derivatives());

    return 0;
}
