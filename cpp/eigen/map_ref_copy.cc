// Per discussion: https://forum.kde.org/viewtopic.php?f=74&t=141703&p=380867

#include <iostream>
#include <Eigen/Dense>

// This adapts Ref<Derived> as RefMap<Derived> to be non-copy-friendly.
// Source taken from Eigen v3.3.3 (git-hg sha: 9e97af7).

namespace Eigen {

// Adapted From: Eigen/src/Core/util/ForwardDeclarations.h
// @note We must preserve all three template parameters for evaluators to stay
// in sync with Ref<>.
template<
    typename PlainObjectType,
    int Options = 0,
    typename StrideType =
        typename internal::conditional<
            PlainObjectType::IsVectorAtCompileTime,
                InnerStride<1>, OuterStride<>>::type>
class RefMap;

namespace internal {

// Adapted From: Eigen/src/Core/Ref.h
template <typename PlainObjectType, int Options, typename StrideType>
struct traits<RefMap<PlainObjectType, Options, StrideType>>
    : public traits<Ref<PlainObjectType, Options, StrideType>> {};

// Adapted From: Eigen/src/Core/CoreEvaluators.h
template<typename PlainObjectType, int RefOptions, typename StrideType> 
struct evaluator<RefMap<PlainObjectType, RefOptions, StrideType> >
  : public mapbase_evaluator<RefMap<PlainObjectType, RefOptions, StrideType>, PlainObjectType>
{
  typedef RefMap<PlainObjectType, RefOptions, StrideType> XprType;
  
  enum {
    Flags = evaluator<Map<PlainObjectType, RefOptions, StrideType> >::Flags,
    Alignment = evaluator<Map<PlainObjectType, RefOptions, StrideType> >::Alignment
  };

  EIGEN_DEVICE_FUNC explicit evaluator(const XprType& ref)
    : mapbase_evaluator<XprType, PlainObjectType>(ref) 
  { }
};

}  // namespace internal

// Adapted From: Eigen/src/Core/Ref.h
/**
 * Provides Ref<Derived> semantics even for Ref<const Derived> cases.
 *
 * Eigen has a specialization of Ref<const Derived> to permit it implicitly copy
 * storage-incompatible matrices and store a local copy the intended Derived
 * type.
 * This class, on the other hand, will throw an error rather than permit
 * a copy to be created, such that it ensures that always refers back to the
 * original data.
 *
 * @ref https://forum.kde.org/viewtopic.php?f=74&t=141703
 */
template<typename PlainObjectType, int Options, typename StrideType> class RefMap
  : public RefBase<RefMap<PlainObjectType, Options, StrideType> >
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
      // We actually do want this to happen, so disable this caveat.
      // EIGEN_STATIC_ASSERT(bool(internal::is_lvalue<Derived>::value), THIS_EXPRESSION_IS_NOT_A_LVALUE__IT_IS_READ_ONLY);
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


  auto A_r = A.row(0);
  auto Ac_r = Ac.row(0);

  cout << PRINT(A_r);

  // Ref<Vector3d> A_ref(A_r);  // Fails as expected.
  // These induce a copy.
  Ref<const Vector3d> A_rt_cref(A_r.transpose());
  cout << PRINT(A_rt_cref);

  Ref<const Matrix<double, 1, 3, RowMajor>> Ac_r_cref_row(Ac_r);
  cout << PRINT(Ac_r_cref_row);

  // Works.
  RefMap<Vector3d> A_c_refmap(A.col(0));
  EVAL(A_c_refmap *= 10);
  cout << PRINT(A_c_refmap);
  cout << PRINT(A);

  RefMap<const Vector3d> A_c_crefmap(A.col(0));
  // A_c_crefmap *= 10;  // Fails as expected.
  cout << PRINT(A_c_crefmap);

  // // Fails as expected.
  // RefMap<Vector3d> A_r_refmap(A.row(0));
  // cout << PRINT(A_r_refmap);

  // // Fails as expected!
  // RefMap<const Vector3d> A_rt_crefmap(A_r.transpose());
  // cout << PRINT(A_rt_crefmap);


  // // This does not work: Does not provide the same stride access.
  // RefMap<RowVector3d> A_r_refmap(A.block(0, 0, 1, 3));

  RefMap<MatrixXd> A_r_refmap_x(A.block(0, 0, 1, 3));

  cout
    << PRINT(A.block(0, 0, 1, 3))
    // << PRINT(A_r_refmap)
    << PRINT(A_r_refmap_x);


  // Check interop between RefMap<> and Ref<>.
  RefMap<Vector3d> A_refmap(A.col(0));
  RefMap<const Vector3d> Ac_crefmap(Ac.col(0));

  Ref<Vector3d> A_ref_2(A_refmap);
  Ref<const Vector3d> Ac_cref_2(Ac_crefmap);

  EVAL(A_ref_2.array() += 2);

  std::cout
    << PRINT(A)
    << PRINT(A_refmap)
    << PRINT(A_ref_2)
    << PRINT(Ac_crefmap)
    << PRINT(Ac_cref_2);

  EVAL(A *= 3);
  std::cout
    << PRINT(A)
    << PRINT(A_r)
    << PRINT(A_refmap)
    << PRINT(Ac_crefmap);

  return 0;
}
