#include <iostream>

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// See: https://forum.kde.org/viewtopic.php?f=74&t=108033

int main() {
  VectorXd eigs_expected(4);
  eigs_expected << 10, -5, -10, 5;
  MatrixXd E = eigs_expected.asDiagonal();

  VectorXd eigs = E.eigenvalues().real();
  cout << "EigenSolver: " << eigs.transpose() << endl;
  /* EigenSolver:  10  -5 -10   5 */
  // Find minimum value.
  int index{};
  eigs.minCoeff(&index);
  cout << "min: " << index << endl;
  /* min: 2 */

  VectorXd eigs_sa = SelfAdjointEigenSolver<MatrixXd>(E).eigenvalues();
  cout << "SelfAdjointEigenSolver: " << eigs_sa.transpose() << endl;
  /* SelfAdjointEigenSolver: -10  -5   5  10 */


  return 0;
}

/*
MATLAB:

e = [10, -5, -10, 5]';
eig(diag(e))'

ans =

   -10    -5     5    10

*/
