#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main() {
  VectorXd X(1);
  X.setZero();

  Ref<VectorXd> X_ref = X;
  X_ref.resize(1);
  // X_ref.resize(2);  // Throws at runtime as expected.

  return 0;
}
