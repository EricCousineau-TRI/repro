#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

int main() {
  MatrixXd X(2, 2);
  X << 1, 2, 3, 4;

  // Ref<MatrixXd> x_row0(X.row(0));  // FAIL
  // Ref<VectorXd> x_row0(X.row(0));  // FAIL
  // Ref<RowVectorXd> x_row0(X.row(0));  // FAIL
  // Ref<MatrixXd::RowXpr> x_row0(X.row(0));  // FAIL
  // Ref<MatrixXd::RowXpr::PlainMatrix> x_row0(X.row(0));  // FAIL
  Ref<MatrixXd> x_row0(X.row(0).transpose());

  // Ref<MatrixXd> x_row0(X.block(0, 0, 1, X.cols()));  // Works
  // Ref<RowVectorXd, 0, InnerStride<>> x_row0(X.row(0));  // Works

  x_row0 *= 10;

  cout << "X: " << X << std::endl;
  cout << "x_row0: " << x_row0 << std::endl;

  return 0;
}
