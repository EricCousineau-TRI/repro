// Per discussion: https://forum.kde.org/viewtopic.php?f=74&t=141703&p=380867

#include <iostream>
#include <Eigen/Dense>

namespace Eigen {

template <typename PlainObjectType>
class RefMap;

namespace internal {

// This should be fine because there are no special traits for Ref<const T>
template <typename PlainObjectType>
struct traits<RefMap<PlainObjectType>>
    : public traits<Ref<PlainObjectType>> {};

}  // namespace internal

template <typename PlainObjectType>
class RefMap : public Eigen::RefBase<RefMap<PlainObjectType>> {
 public:
  typedef Eigen::RefBase<RefMap<PlainObjectType>> Base;

  template <typename PlainObjectTypeInF>
  RefMap(PlainObjectTypeInF&& expr) {
    typedef std::decay_t<decltype(expr.derived())> Derived;
    typedef internal::traits<Ref<Derived>> Traits;
    static_assert(
      Traits::template match<Derived>::MatchAtCompileTime,
      "STORAGE_LAYOUT_DOES_NOT_MATCH");
    static_assert(
      !Derived::IsPlainObjectBase,
      "BAD_F00D");
    Base::construct(expr.const_cast_derived());

    if (expr.size() > 0) {
      // Ensure that we have properly strided data
      // E.g., guard against getting the nested expression data / strides in
      // a transpose() expression.
      const int last = expr.size() - 1;
      eigen_assert(
        &this->coeffRef(0) == &expr.coeffRef(0) &&
        &this->coeffRef(last) == &expr.coeffRef(last) &&
        "ERROR: Data and stride for input object (PlainObjectTypeInF) do not \
match those of template parameter (PlainObjectType).");
    }
  }
};

}  // namespace Eigen


#define EVAL(x) std::cout << ">>> " #x ";" << std::endl; x; std::cout << std::endl
#define PRINT(x) ">>> " #x << std::endl << (x) << std::endl << std::endl

int main() {
  using namespace Eigen;
  using std::cout;
  using std::endl;

  MatrixXd A(3, 3);
  const MatrixXd& Ac(A);
  A << 1, 2, 3,
       4, 5, 6,
       7, 8, 9;
  cout << PRINT(A);

  Ref<MatrixXd> A_block = A.block(1, 1, 2, 2);
  EVAL(A *= 2);
  cout
    << PRINT(A)
    << PRINT(A_block);


  auto A_rt = A.row(0).transpose();
  auto Ac_rt = Ac.row(0).transpose();

  cout << PRINT(A_rt.transpose());

  // Ref<Vector3d> A_ref(A_rt);  // Fails as expected.
  // These induce a copy.
  Ref<const Vector3d> A_cref(A_rt);
  cout << PRINT(A_cref.transpose());

  Ref<const Matrix<double, 1, 3, RowMajor>> A_cref_row(Ac_rt);
  cout << PRINT(A_cref_row.transpose());

  RefMap<Vector3d> A_c_refmap(A.col(0));
  cout << PRINT(A_c_refmap.transpose());

  // Will fail.
  RefMap<RowVector3d> A_r_refmap(A.row(0));
  cout << PRINT(A_r_refmap);

  // Will also fail.
  RefMap<Vector3d> A_refmap(A_rt);
  RefMap<const Vector3d> Ac_crefmap(Ac_rt);

  RefMap<const Vector3d> A_crefmap(A_rt);
  // RefMap<Vector3d> Ac_refmap(Ac_rt);  // Fails as expected.
  cout << PRINT(Ac_crefmap.transpose());

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
