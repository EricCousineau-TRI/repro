// @ref https://eigen.tuxfamily.org/dox/classEigen_1_1SparseMatrixBase.html#a3c6eeff61503ef26a6e9f830feecec08

#include <Eigen/Core>
#include <iostream>
using namespace Eigen;
using namespace std;
// define a custom template unary functor
template<typename Scalar>
struct CwiseClampOp {
  CwiseClampOp(const Scalar& inf, const Scalar& sup)
      : m_inf(inf), m_sup(sup) {}
  const Scalar operator()(const Scalar& x) const {
    return x<m_inf ? m_inf : (x>m_sup ? m_sup : x);
  }
  Scalar m_inf, m_sup;
};

template<typename Scalar>
struct CwiseClampBoolOp {
  CwiseClampBoolOp(const Scalar& inf, const Scalar& sup)
      : m_inf(inf), m_sup(sup) {}
  bool operator()(const Scalar& x) const {
    return x<m_inf || x>m_sup;
  }
  Scalar m_inf, m_sup;
};

struct Value {
  double a{1.5};
  int b{2};
};

/*
Look at:
  Eigen::...
    scalar_real_ref_op
    real_ref_impl
      numext::real_ref, etc.
  These seem to be able to return write-able values.
    Need to see how to overload the "run()" method.
*/
template <typename Scalar, typename FieldType>
class CwiseFieldExtractOp {
 public:
  typedef FieldType Scalar::* FieldPtr;
  CwiseFieldExtractOp(FieldPtr member)
      : member_(member) {}
  FieldType& operator() (Scalar& x) const { return x.*member_; }
  const FieldType& operator() (const Scalar& x) const { return x.*member_; }
 private:
  FieldPtr member_{};
};

template <typename Scalar, typename FieldType>
auto extract_field_op(FieldType Scalar::* member) {
  return CwiseFieldExtractOp<Scalar, FieldType>(member);
}

template <typename XprType, typename FieldPtr>
auto view_field(XprType&& X, FieldPtr member) {
  return X.unaryViewExpr(extract_field_op(member));
}

template <typename Scalar>
using MatrixX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

int main(int, char**)
{
  Matrix4d m1 = Matrix4d::Random();
  cout
      << m1 << endl
      << "becomes: " << endl
      << m1.unaryExpr(CwiseClampOp<double>(-0.5,0.5)) << endl
      << "orig: " << endl
      << m1 << endl
      << "bool: " << endl
      << m1.unaryExpr(CwiseClampBoolOp<double>(-0.5, 0.5)) << endl;

  typedef MatrixX<Value> MatrixXV;
  MatrixXV v1(2, 2);
  cout
      << "a: " << endl
      << view_field(v1, &Value::b) << endl
      << "b: " << endl
      << view_field(v1, &Value::b) << endl;
  // // TODO(eric.cousineau): Figure this part out.
  // view_field(v1, &Value::b).setConstant(10);
  return 0;
}
