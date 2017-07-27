#include <iostream>
#include <string>
#include <type_traits>

using namespace std;

struct A {
  int a;
  double b;
};

struct B {
  string x;
  int y;
};

int main() {
  cout << std::is_pod<A>::value << endl;
  cout << std::is_pod<B>::value << endl;
  return 0;
}
