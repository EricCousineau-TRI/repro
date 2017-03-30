#include <iostream>
#include <Eigen/Dense>

using Eigen::VectorXd;
using Eigen::MatrixXd;

int main() {
  MatrixXd G(3, 3);
  G <<
    1, 0, 0,
    0, 3, 0,
    0, 0, 2;
  VectorXd c(3);
  c << -1, -6, -2;
  VectorXd x(3);
  x << 1, 2, 1;

  double optimal_cost = 0.5 * x.dot(G * x + c);
  Eigen::VectorXd Gxc = G * x + c;
  std::cout << "cost: " << optimal_cost
    << "\nG: " << G
    << "\nc: " << c.transpose()
    << "\nx: " << x.transpose()
    << "\nG * x: " << (G * x).transpose()
    << "\nG * x + c: " << (G * x + c).transpose()
    << "\nGxc: " << Gxc.transpose()
    << "\nx + c: " << (x + c).transpose()
    << "\n---\n";
  return 0;
}
