// Per discussion: https://forum.kde.org/viewtopic.php?f=74&t=141703&p=380867

#include <iostream>
#include <Eigen/Dense>

#define EVAL(x) std::cout << ">>> " #x ";" << std::endl; x; std::cout << std::endl
#define PRINT(x) ">>> " #x << std::endl << (x) << std::endl << std::endl

template <typename PlainObjectType>
class RefMap : public Eigen::Map<PlainObjectType> {
 public:
  typedef Eigen::Map<PlainObjectType> Base;

  template <typename PlainObjectTypeIn>
  RefMap(PlainObjectTypeIn&& in)
    : Base(in.data(), in.rows(), in.cols()) {
    std::cout << "stride: (" << in.outerStride()
      << ", " << in.innerStride() << ")" << std::endl;
  }
};

int main() {
  using namespace Eigen;

  MatrixXd A(3, 3);
  const MatrixXd& Ac(A);
  A << 1, 2, 3,
       4, 5, 6,
       7, 8, 9;

  auto A_rt = A.row(0).transpose();
  auto Ac_rt = Ac.row(0).transpose();

  // Map<Vector3d> A_refmap(A);
  RefMap<Vector3d> A_refmap(A_rt);
  RefMap<const Vector3d> Ac_crefmap(Ac_rt);

  RefMap<const Vector3d> A_crefmap(A_rt);
  // RefMap<Vector3d> Ac_refmap(Ac_rt);  // Fails as expected.

  std::cout
    << PRINT(A_rt.transpose())
    << PRINT(A_refmap.transpose())
    << PRINT(Ac_crefmap.transpose());

  // This induces a copy.
  Ref<const Vector3d> A_cref(A_rt);
  Ref<const Matrix<double, 1, 3, RowMajor>> A_cref_row(Ac_rt);

  std::cout << "---\n";
  EVAL(A *= 3);
  std::cout  
    << PRINT(A_rt.transpose())
    << PRINT(A_refmap.transpose())
    << PRINT(A_crefmap.transpose())
    << PRINT(A_cref.transpose())
    << PRINT(A_cref_row);

  return 0;
}
