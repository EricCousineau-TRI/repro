#include <iostream>
#include <Eigen/Dense>

using std::cout;
using std::endl;

int main() {
    Eigen::VectorXd x(2);
    x << 1., 2.;
    // max(1.5).array().
    Eigen::VectorXd y = x.array().max(1.5).max(10 * x.array()).max(15);
    cout << "y: " << y << endl;

    Eigen::Matrix3Xd z(3, 2);
    z <<
      1, 10,
      8, 4,
      3, 3;

    Eigen::VectorXd z1, z2;
    z1 = z.rowwise().maxCoeff();
    z2 = z.rowwise().minCoeff();

    cout << "z1: " << z1.transpose() << "\n"
      << "z2: " << z2.transpose() << "\n";

    return 0;
}
