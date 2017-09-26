// Per discussion: https://forum.kde.org/viewtopic.php?f=74&t=141703&p=380867

#include <iostream>
#include <Eigen/Dense>

template <typename PlainObjectType>
class RefMap : public Eigen::Map<PlainObjectType> {
 public:
  typedef Eigen::Map<PlainObjectType> Base;

  template <typename PlainObjectTypeIn>
  RefMap(PlainObjectTypeIn&& in)
    : Base(in.data(), in.rows(), in.cols()) {}
};

int main() {
  using namespace Eigen;

  VectorXd X(3);
  const VectorXd& X_c(X);
  X << 1, 2, 3;

  // Map<Vector3d> Y(X);
  RefMap<Vector3d> Y(X);
  RefMap<const Vector3d> Y_c(X);

  // RefMap<Vector3d> Yc(X_c);  // Fails as expected.
  RefMap<const Vector3d> Yc_c(X_c);

  std::cout << Y.transpose() << std::endl;

  // Try to induce a copy.
  Ref<const Vector3d> Z_c(X);
  Ref<Matrix<double, 1, 3, RowMajor>> A_c(X.transpose());

  std::cout << A_c << std::endl;

  X *= 3;

  std::cout << "---\n" << Z_c.transpose() << "\n" << A_c << "\n";

  return 0;
}
