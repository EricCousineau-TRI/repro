#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// Check if we can re-assign structs with refs.
struct A {
  Ref<VectorXd> X;
};

int main() {
  VectorXd X(1);
  X.setZero();

  Ref<VectorXd> X_ref = X;
  X_ref.resize(1);
  // X_ref.resize(2);  // Throws at runtime as expected.

  // VectorXd Y(5);
  // Ref<VectorXd> Y_head = Y.head(3);
  // Ref<VectorXd> Y_tail = Y.tail(2);
  // A a {Y_head};
  // A b {Y_tail};
  // a = b;

  return 0;
}
