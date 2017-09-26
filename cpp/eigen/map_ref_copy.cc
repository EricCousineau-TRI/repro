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

  MatrixXd X(3, 3);
  const MatrixXd& X_c(X);
  X << 1, 2, 3,
       4, 5, 6,
       7, 8, 9;

  auto Xr = X.row(0).transpose();
  auto X_cr = X_c.row(0).transpose();

  // Map<Vector3d> Y(X);
  RefMap<Vector3d> Y(Xr);
  RefMap<const Vector3d> Y_c(Xr);

  // RefMap<Vector3d> Yc(X_cr);  // Fails as expected.
  RefMap<const Vector3d> Yc_c(X_cr);

  std::cout << Y.transpose() << std::endl;

  // Try to induce a copy.
  Ref<const Vector3d> Z_c(Xr);
  Ref<const Matrix<double, 1, 3, RowMajor>> A_c(X_cr);

  std::cout << A_c << std::endl;

  X *= 3;

  std::cout
      << "---\n"
      << Y.transpose() << "\n"
      << Y_c.transpose() << "\n"
      << Xr.transpose() << "\n"
      << Z_c.transpose() << "\n"
      << A_c << "\n";

  return 0;
}
