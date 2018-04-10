#include <iostream>

#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>

using std::cout;
using std::endl;
using Eigen::VectorXd;

using AutoDiffXd = Eigen::AutoDiffScalar<Eigen::VectorXd>;

template <typename T>
using storage_for = typename std::aligned_storage<sizeof(T), alignof(T)>::type;

int main() {
  constexpr int n = sizeof(AutoDiffXd);
  storage_for<AutoDiffXd> buffer;
  AutoDiffXd* value = reinterpret_cast<AutoDiffXd*>(&buffer);
  memset(&buffer, 0, n);
  // new (value) VectorXd(0);
  *value = AutoDiffXd(10, VectorXd::Constant(3, 10));
  cout
      << "value: " << *value << " - "
      << value->derivatives().transpose() << endl;
  // delete value;
  value->~AutoDiffXd();
  return 0;
}
