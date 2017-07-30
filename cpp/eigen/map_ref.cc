// Purpose: Determine if it is possible to mask an Eigen::Map<> via
// Eigen::Ref<>.
#include <iostream>
#include <memory>

#include <Eigen/Core>

using namespace std;
using namespace Eigen;

MatrixXd copy(Ref<const MatrixXd> x) {
  return x;
}

int main() {
  array<double, 4> x_raw {{1, 2, 3, 4}};
  Map<MatrixXd> x_map(x_raw.data(), 2, 2);

  MatrixXd y = copy(x_map);
  cout << y << endl;

  return 0;
}
