// TODO: See if view is compatible with Eigen::Ref<MatrixX<Scalar>>

#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

struct Value {
  double a{1.5};
  int b{2};
};

// See: CommonCwiseUnaryOps.h for using the trait types (NonConstRealReturnType, Matrix::real())
// See: MathFunctions.h for registration (numext::real_ref())
/*
// From operator, real_ref
template<typename Scalar>
struct scalar_real_ref_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_real_ref_op)
  typedef typename NumTraits<Scalar>::Real result_type;
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE result_type& operator() (const Scalar& a) const { return numext::real_ref(*const_cast<Scalar*>(&a)); }
};
template<typename Scalar>
struct functor_traits<scalar_real_ref_op<Scalar> >
{ enum { Cost = 0, PacketAccess = false }; };
*/

struct get_a {
  double operator()(const Value& v) const {
    return v.a;
  }
};

// Savage. Use aforemention hack from scalar_real_ref_op.
struct get_mutable_a_direct {
  double& operator()(const Value& v) const {
    // 
    return const_cast<Value&>(v).a;
  }
};

// For decltype only. Will cause linker error if called.
template <typename Derived>
Derived derived_of(const Eigen::MatrixBase<Derived>&);

template <typename Op, typename XprType>
struct unary_mutable_helper {
  typedef decltype(derived_of(std::declval<XprType>())) Derived;
  typedef typename Derived::Scalar Scalar;
  // typedef decltype(op(std::declval<const Scalar&>())) const_return_type;
  // typedef decltype(op(std::declval<Scalar&&>())) rvalue_return_type;
  static auto run(XprType& xpr, const Op& op) {
    // lvalue
    typedef decltype(op(std::declval<Scalar&>())) lvalue_return_type;
    auto wrap_const_cast = [&op](const Scalar& scalar) -> lvalue_return_type {
      return op(const_cast<Scalar&>(scalar));
    };
    typedef Eigen::CwiseUnaryView<decltype(wrap_const_cast), Derived> NonConstView;
    return NonConstView(xpr, wrap_const_cast);
  }
};

// Keep savage nature contained.
template <typename Op, typename XprType>
auto unaryMutableExpr(XprType&& xpr, const Op& op = Op()) {
  return unary_mutable_helper<Op, XprType>::run(xpr, op);
}

// Follow drake/common/eigen_types.h
template <typename Scalar>
using MatrixX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

struct get_mutable_a {
  double& operator()(Value& v) const {
    return v.a;
  }
  const double& operator()(const Value& v) const {
    return v.a;
  }
};

int main() {
  MatrixX<Value> X(2, 2);
  auto X_ac = X.unaryViewExpr(get_a());
  cout << X_ac << endl;
  // Savage.
  auto X_am_direct = Eigen::CwiseUnaryView<get_mutable_a_direct, MatrixX<Value>>(
      X, get_mutable_a_direct());
  X_am_direct.setConstant(20);
  cout << X_ac << endl;
  // Less? savage.
  auto X_am = unaryMutableExpr(X, get_mutable_a());
  X_am *= 10;
  cout << X_ac << endl;

  // // Fails as desired.
  // const auto& Xc = X;
  // auto Xc_am = unaryMutableExpr(Xc, get_mutable_a());

  return 0;
}
