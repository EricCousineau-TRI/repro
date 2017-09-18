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
  Eigen::VectorXd X(3);
  const Eigen::VectorXd& X_c(X);
  X << 1, 2, 3;

  // Eigen::Map<Eigen::Vector3d> Y(X);
  RefMap<Eigen::Vector3d> Y(X);
  RefMap<const Eigen::Vector3d> Y_c(X);

  // RefMap<Eigen::Vector3d> Yc(X_c);  // Fails as expected.
  RefMap<const Eigen::Vector3d> Yc_c(X_c);


  std::cout << Y.transpose() << std::endl;

  return 0;
}
