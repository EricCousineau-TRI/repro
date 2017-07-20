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
    return const_cast<Value&>(v).a;
  }
};

// For decltype only. Will cause linker error if called.
template <typename Derived>
Derived derived_of(const Eigen::MatrixBase<Derived>&);

template <typename Op, typename XprType>
struct unary_mutable_helper {
  typedef decltype(derived_of(std::declval<XprType>())) Derived; // Is this needed???
  typedef typename Derived::Scalar Scalar;

  static auto run(XprType& xpr, const Op& op) {
    // lvalue
    typedef decltype(op(std::declval<Scalar&>())) lvalue_return_type;
    auto wrap_const_cast = [&op](const Scalar& scalar) -> lvalue_return_type {
      return op(const_cast<Scalar&>(scalar));
    };
    typedef Eigen::CwiseUnaryView<decltype(wrap_const_cast), Derived> NonConstView;
    return NonConstView(xpr, wrap_const_cast);
  }
  // Don't worry about move semantics until properly supported?
  
  static auto run(const XprType& xpr, const Op& op) {
    // Use the good stuff.
    return xpr.unaryViewExpr(op);
  }
};

// Keep savage nature contained.
template <typename Op, typename XprType>
auto unaryExprFlex(XprType&& xpr, const Op& op) {
  typedef std::decay_t<XprType> XprTypeBare;
  return unary_mutable_helper<Op, XprTypeBare>::run(
      std::forward<XprType>(xpr), op);
}

// Follow drake/common/eigen_types.h
template <typename Scalar>
using MatrixX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

struct get_a_flex {
  double& operator()(Value& v) const {
    return v.a;
  }
  const double& operator()(const Value& v) const {
    return v.a;
  }
};

// Simple helper to extract a given field.
template <typename Scalar, typename FieldType>
class extract_field {
 public:
  typedef FieldType Scalar::* FieldPtr;
  extract_field() = delete;
  extract_field(FieldPtr member)
      : member_(member) {}
  FieldType& operator() (Scalar& x) const { return x.*member_; }
  const FieldType& operator() (const Scalar& x) const { return x.*member_; }
 private:
  FieldPtr member_{};
};

template <typename XprType, typename Scalar, typename FieldType>
auto unaryFieldExpr(XprType&& xpr, FieldType Scalar::* member) {
  extract_field<Scalar, FieldType> op(member);
  return unaryExprFlex(std::forward<XprType>(xpr), op);
}

int main() {
  MatrixX<Value> X(2, 2);
  auto X_ac = X.unaryViewExpr(get_a());
  cout << X_ac << endl;
  
  // Savage. But direct.
  auto X_am_direct = Eigen::CwiseUnaryView<get_mutable_a_direct, MatrixX<Value>>(
      X, get_mutable_a_direct());
  X_am_direct.setConstant(20);
  cout << X_ac << endl;

  // Less? savage.
  auto X_am = unaryExprFlex(X, get_a_flex());
  X_am *= 10;
  cout << X_ac << endl;

  // Works.
  const auto& Xc = X;
  auto Xc_am = unaryExprFlex(Xc, get_a_flex());
  // Xc_am.setConstant(20);  // Fails as desired.
  cout << Xc_am << endl;

  auto X_bm = unaryExprFlex(X, [](Value& v) -> auto& { return v.b; });
  cout << X_bm << endl;
  X_bm.setConstant(10);
  cout << X_bm << endl;

  // // Refs don't work :(
  // Eigen::Ref<Eigen::MatrixXd> Xref = X_am;

  auto X_bmf = unaryFieldExpr(X, &Value::b);
  cout << X_bmf << endl;

  return 0;
}
