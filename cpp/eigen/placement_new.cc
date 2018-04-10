#include <iostream>

#include <Eigen/Dense>

using std::cout;
using std::endl;
using Eigen::VectorXd;

template <typename T>
using storage_for = typename std::aligned_storage<sizeof(T), alignof(T)>::type;

int main() {
  constexpr int n = sizeof(VectorXd);
  storage_for<VectorXd> buffer;
  VectorXd* value = reinterpret_cast<VectorXd*>(&buffer);
  new (value) VectorXd(0);
  *value = VectorXd::Constant(3, 10);
  cout << "value: " << value->transpose() << endl;
  // delete value;
  value->~VectorXd();
  return 0;
}
