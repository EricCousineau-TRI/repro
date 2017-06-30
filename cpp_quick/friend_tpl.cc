#include <iostream>
using namespace std;

class A {
 public:
  int x() { return x_; }
 private:
  int x_{0};

  template <char TestName[]>
  friend class TestHelper;
};

// Would need to be declared somewhere common.
template <char TestName[]>
class TestFriend {};


// In the testing code
constexpr char test_name[] = "MyTest";
template <>
class TestFriend<test_name> {
 public:
  void set_x(A* a) {
    a->x_ = 1;
  }
};

int main() {
  A a;
  cout << a.x() << endl;

  TestFriend<test_name> helper;
  helper.set_x(&a);

  cout << a.x() << endl;

  return 0;
}
