#include <iostream>

#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

#define PRINT(x) #x ": " << (x) << endl

using ArrayXb = Array<bool, Dynamic, 1>;

int main() {
  ArrayXb empty;
  ArrayXb size1_true(1);
  size1_true << true;

  cout
      << PRINT(empty.all())
      << PRINT(size1_true.all());
  return 0;
}
