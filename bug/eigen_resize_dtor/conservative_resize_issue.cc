#define EIGEN_DONT_INLINE 1

#include <iostream>
#include <string>
#include <utility>

#include <Eigen/Core>

using std::cout;
using std::endl;
using std::string;

using Scalar = string;
using VectorXs = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

void AppendToVector(const Scalar& s, VectorXs* px) {
  VectorXs& derived = *px;
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

    VectorXs a(2);
    a << "x", "y";
    AppendToVector("z", &a);

    return 0;
}
