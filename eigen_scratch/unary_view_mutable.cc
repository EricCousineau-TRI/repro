#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

struct Value {
  double a{1.5};
  int b{2};
};

struct get_a {
  double operator()(const Value& v) const {
    return v.a;
  }
};

// Follow drake/common/eigen_types.h
template <typename Scalar>
using MatrixX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

int main() {
  MatrixX<Value> X(2, 2);
  cout
    << X.unaryViewExpr(get_a()) << endl;
  return 0;
}
