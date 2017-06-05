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

  return 0;
}
