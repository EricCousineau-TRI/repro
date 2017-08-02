/*
From Mmanu:
https://gist.github.com/m-chaturvedi/aed94a7fa4b51d5bfcf8ea5f063790dc

Execute in Bash:

(
  set -e -u
  mkdir -p build/
  name=ubsan_zero_size_mult
  # eigen_INCLUDE_DIR=/usr/include/eigen3
  eigen_INCLUDE_DIR=../../externals/eigen
  clang++-3.9 -g -std=c++11 \
    -I ${eigen_INCLUDE_DIR} \
    -fsanitize=undefined -fno-sanitize-recover=undefined \
    -fsanitize-trap=undefined \
    ${name}.cc -o build/${name}
  gdb -ex run ./build/${name}
)

Error seems to come from here:
  (gdb) f 1
  #1  0x000000000046d2ca in Eigen::internal::triangular_solver_selector<...>::run (
      lhs=Eigen::Matrix<double, 2, 2, ColMajor> (data ptr: 0x6b6c50)
          1 0 
          0 1 , rhs=Eigen::Matrix<double, 2, 0, ColMajor> (data ptr: 0x0)

          ) at /usr/include/eigen3/Eigen/src/Core/SolveTriangular.h:102
  102           ::run(size, othersize, &actualLhs.coeffRef(0,0), actualLhs.outerStride(), &rhs.coeffRef(0,0), rhs.outerStride(), blocking);

rhs.coeffRef(0,0) is invalid in this case.
Would `rhs.data()` not work?

*/
#include <bits/stdc++.h>
#include <Eigen/Core>
#include <Eigen/Cholesky>
using namespace std;
using namespace Eigen;

int main() {
  MatrixXd A(2, 2);
  A << 1, 0, 0, 1;
  MatrixXd B = MatrixXd::Zero(2,0);
  LLT<MatrixXd> llt(A);
  MatrixXd X = llt.solve(B);
}
