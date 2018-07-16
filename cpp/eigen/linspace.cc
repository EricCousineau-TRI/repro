#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

int main() {
  int n = 10;
  cout << VectorXi::LinSpaced(n, 0, n - 1).transpose() << endl;
  return 0;
}
