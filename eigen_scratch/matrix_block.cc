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

template<typename Derived>
void fill(DenseBase<Derived>& x) {
    x.setConstant(1);
}

int main() {
    MatrixXd A(2, 2);
    Matrix3d B;

    fill(A);
    fill(B);

    cout
        << "A: " << endl << A << endl << endl
        << "B: " << endl << B << endl << endl;

    return 0;
}
