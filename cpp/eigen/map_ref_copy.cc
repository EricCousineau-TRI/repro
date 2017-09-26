// Per discussion: https://forum.kde.org/viewtopic.php?f=74&t=141703&p=380867

#include <iostream>
#include <Eigen/Dense>

namespace Eigen {

// Eigen/src/Core/util/ForwardDeclarations.h
template<typename PlainObjectType>
class RefMap;

namespace internal {

// Eigen/src/Core/CoreEvaluators.h
template<typename PlainObjectType> 
struct evaluator<RefMap<PlainObjectType> >
  : public mapbase_evaluator<RefMap<PlainObjectType>, PlainObjectType>
{
  typedef RefMap<PlainObjectType> XprType;
  
  enum {
    Flags = evaluator<Map<PlainObjectType> >::Flags,
    Alignment = evaluator<Map<PlainObjectType> >::Alignment
  };

  EIGEN_DEVICE_FUNC explicit evaluator(const XprType& ref)
    : mapbase_evaluator<XprType, PlainObjectType>(ref) 
  { }
};

// Eigen/src/Core/Ref.h
template <typename PlainObjectType>
struct traits<RefMap<PlainObjectType>>
    : public traits<Ref<PlainObjectType>> {};

}  // namespace internal

// Eigen/src/Core/Ref.h
template<typename PlainObjectType> class RefMap
  : public RefBase<RefMap<PlainObjectType> >
{
  private:
    typedef internal::traits<RefMap> Traits;
    template<typename Derived>
    EIGEN_DEVICE_FUNC inline RefMap(const PlainObjectBase<Derived>& expr,
                                 typename internal::enable_if<bool(Traits::template match<Derived>::MatchAtCompileTime),Derived>::type* = 0);
  public:

    typedef RefBase<RefMap> Base;
    EIGEN_DENSE_PUBLIC_INTERFACE(RefMap)

    template<typename Derived>
    EIGEN_DEVICE_FUNC inline RefMap(PlainObjectBase<Derived>& expr,
                                 typename internal::enable_if<bool(Traits::template match<Derived>::MatchAtCompileTime),Derived>::type* = 0)
    {
      EIGEN_STATIC_ASSERT(bool(Traits::template match<Derived>::MatchAtCompileTime), STORAGE_LAYOUT_DOES_NOT_MATCH);
      Base::construct(expr.derived());
    }
    template<typename Derived>
    EIGEN_DEVICE_FUNC inline RefMap(const DenseBase<Derived>& expr,
                                 typename internal::enable_if<bool(Traits::template match<Derived>::MatchAtCompileTime),Derived>::type* = 0)
    {
      EIGEN_STATIC_ASSERT(bool(internal::is_lvalue<Derived>::value), THIS_EXPRESSION_IS_NOT_A_LVALUE__IT_IS_READ_ONLY);
      EIGEN_STATIC_ASSERT(bool(Traits::template match<Derived>::MatchAtCompileTime), STORAGE_LAYOUT_DOES_NOT_MATCH);
      EIGEN_STATIC_ASSERT(!Derived::IsPlainObjectBase,THIS_EXPRESSION_IS_NOT_A_LVALUE__IT_IS_READ_ONLY);
      Base::construct(expr.const_cast_derived());
    }

    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(RefMap)
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

  Ref<MatrixXd> A_block_ref = A.block(1, 1, 2, 2);
  RefMap<MatrixXd> A_block_refmap = A.block(1, 1, 2, 2);
  EVAL(A *= 2);
  cout
    << PRINT(A)
    << PRINT(A_block_ref)
    << PRINT(A_block_refmap);


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

  // // Will fail.
  // RefMap<RowVector3d> A_r_refmap(A.row(0));
  // cout << PRINT(A_r_refmap);

  // // Will also fail.
  // RefMap<Vector3d> A_refmap(A_rt);
  // RefMap<const Vector3d> Ac_crefmap(Ac_rt);

  // RefMap<const Vector3d> A_crefmap(A_rt);
  // // RefMap<Vector3d> Ac_refmap(Ac_rt);  // Fails as expected.
  // cout << PRINT(Ac_crefmap.transpose());

  // std::cout << "---\n";
  // EVAL(A *= 3);
  // std::cout  
  //   << PRINT(A_rt.transpose())
  //   << PRINT(A_refmap.transpose())
  //   << PRINT(A_crefmap.transpose())
  //   << PRINT(A_cref.transpose())
  //   << PRINT(A_cref_row);

  return 0;
}
