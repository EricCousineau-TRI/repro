#include <iostream>

using namespace std;

struct A {
  // Should enable an error with -Wshadow.
  int x;
  A(int x)
    : x(x) {}
};

int main() {
  A x(1);
  cout << x.x << endl;
  return 0;
}
