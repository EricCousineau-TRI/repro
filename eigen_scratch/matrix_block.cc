// Purpose: Figure out how to generically use different expressions, to be compatible with assigning to blocks and what not

#include "cpp_quick/name_trait.h"

#include <string>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>

using std::cout;
using std::endl;
using std::string;

using Eigen::DenseBase;
using Eigen::Block;
using Eigen::MatrixXd;
using Eigen::Matrix3d;
using Eigen::MatrixBase;

// template<typename Derived>
// void fill(MatrixBase<Derived> const& x_hack) {
//     // Be wary!!! Using hack from:
//     // https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html#title3
//     auto& x = const_cast<MatrixBase<Derived>&>(x_hack);
//     x.setConstant(1);
// }

/**
Obtain lvalue reference from rvalue reference.
Only use if you know what you are doing!
*/
template<typename T>
T& to_reference(T&& x) {
    return static_cast<T&>(x);
}

template<typename Derived>
void fill(MatrixBase<Derived>& x) {
    x.setConstant(1);
}
template<typename Derived>
void fill(MatrixBase<Derived>&& x) {
    // Secondary hack (cleaner due to not fiddling with const, but still a hack)
    // Cleaner alternative: Reimplement the functionality
    fill(to_reference(x));
}

Matrix3d example() {
    Matrix3d x;
    x.setConstant(10);
    return x;
}

int main() {
    MatrixXd A(2, 2);
    Matrix3d B;
    MatrixXd C(3, 2);

    fill(A);
    fill(B);
    fill(C.block(0, 0, 2, 2));

    // // This fails
    // fill(C + 5 * MatrixXd::Ones(3, 2));

    // This does not
    fill(example());

    // C.block(0, 0, 2, 2) << A; // This works because Eigen::Block can return a reference to itself

    cout
        << "A: " << endl << A << endl << endl
        << "B: " << endl << B << endl << endl
        << "C: " << endl << C << endl << endl;

    return 0;
}
