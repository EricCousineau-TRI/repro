// Illustration for: https://github.com/RobotLocomotion/drake/pull/11090
#include <iostream>

#include <Eigen/Dense>

using Eigen::Translation3d;
using Eigen::Vector3d;

int main() {
  Translation3d X_AB(1, 2, 3);
  // My scribbling crayon is mighty!
  // N.B. I dunno if this is SO(3). Just guessin'.
  X_AB.linear() <<
    0, 1, 0,
    1, 0, 0,
    0, 0, -1;
  Vector3d p_BC(10, 20, 30);
  std::cout << (X_AB * p_BC).transpose() << std::endl;
  return 0;
}
