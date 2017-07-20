// Goal: Test C++ type affinity to Matrix or Array
#include <iostream>

#include <Eigen/Dense>

using std::cout;
using std::endl;

using Eigen::MatrixBase;
using Eigen::MatrixXd;
using Eigen::ArrayBase;
using Eigen::ArrayXd;

template <typename Derived>
void func_matrix(const MatrixBase<Derived> &x) {
    cout << "matrix" << endl;
}

template <typename Derived>
void func_array(const ArrayBase<Derived> &x) {
    cout << "array" << endl;
}

int main() {
    MatrixXd m(1, 1);
    func_matrix(m);
    // func_array(m); // This will not bind
    ArrayXd a(1, 1);
    // func_matrix(a); // This will not bind
    func_array(a);

    func_matrix(a.matrix());
    func_array(m.array());

    return 0;
}
