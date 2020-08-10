#include <iostream>

struct A {
  A() { }
  A(const A&) { std::cout << "A(A)\n"; }
  // If this were explicit, construction of `A x = 1` would fail.
  /*explicit*/ A(int) { std::cout << "A(int)\n"; }
  A& operator=(int) { std::cout << "A=int\n"; return *this; }
};

int main() {
  A x = 1;
  // A x(1);
  x = 2;
  A y;
  y = 3;
  return 0;
}
