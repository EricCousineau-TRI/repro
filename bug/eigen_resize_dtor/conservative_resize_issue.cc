#define EIGEN_DONT_INLINE

#include <iostream>
#include <string>
#include <utility>

#include <Eigen/Core>

using std::cout;
using std::endl;
using std::string;

using Eigen::VectorXd;

template<typename Derived, typename Scalar,
    typename = typename std::enable_if<Derived::ColsAtCompileTime == 1>::type>
void AppendToVector(const Scalar& s, Eigen::MatrixBase<Derived>* px) {
  Derived& derived = px->derived();
  int initial_size = derived.size();
#ifdef USE_BAD
  // TODO(eric.cousineau): This causes a memory leak?
  derived.conservativeResize(initial_size + 1);
#else
  derived.conservativeResize(initial_size + 1, Eigen::NoChange);
#endif // USE_BAD
  derived(initial_size) = s;
}

int main() {
    using VectorXd = Eigen::VectorXd;
    using VectorXXd = Eigen::Matrix<VectorXd, Eigen::Dynamic, 1>;
    VectorXXd a(2);
//    a << VectorXd::Zero(1), VectorXd::Zero(2);
    a(0) = VectorXd::Zero(1);
    a(1) = VectorXd::Zero(2);
    AppendToVector(VectorXd::Ones(3), &a);

    return 0;
}
