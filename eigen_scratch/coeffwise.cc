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
    return 0;
}
