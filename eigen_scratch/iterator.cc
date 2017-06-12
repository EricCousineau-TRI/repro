#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// TODO(eric.cousineau): Do full implementation for more robust access.
template <typename XprType>
class IterableMatrix {
 public:
  IterableMatrix(XprType&& xpr)
      : xpr_(xpr) {}

  auto begin() { return xpr_.data(); }
  auto begin() const { return xpr_.data(); }

  auto end() { return xpr_.data() + xpr_.size(); }
  auto end() const { return xpr_.data() + xpr_.size(); }

 private:
  XprType xpr_;
};

template <typename XprType>
auto MakeIterableMatrix(XprType&& xpr) {
  return IterableMatrix<XprType>(std::forward<XprType>(xpr));
}

int main() {
  VectorXd x(5);
  x << 1, 2, 3, 4, 5;
  for (auto&& xi : MakeIterableMatrix(x)) {
    cout << xi << " ";
    xi *= 2;
  }
  cout << endl;
  const VectorXd& x_const = x;
  for (auto&& xi : MakeIterableMatrix(x_const)) {
    cout << xi << " ";
    // xi *= 2;  // Will trigger an error as expected.
  }
  cout << endl;
  cout << x.transpose() << endl;
  return 0;
}
