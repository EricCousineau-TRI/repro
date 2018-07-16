#include <iostream>

using std::cout;
using std::endl;

enum Test : int {
  A = 0,
  B = 1,
};

int main() {
  Test x = A;
  Test y = Test::B;

  int x_i = x;
  int y_i = y;

  cout << x_i << " - " << y_i << endl;

  return 0;
}
