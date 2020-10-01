#include <iostream>

namespace nested {

struct C {};

int operator+(C, C) { return 10; }
int operator+(C, int) { return 20; }

}

using nested::C;
using nested::operator+;

typedef int (*MyFunc)(C, C);

int main() {
  std::cout << (C{} + C{}) << std::endl;

  MyFunc func1 = &operator+;
  auto func2 = static_cast<int (*)(C, int)>(&nested::operator+);
  auto func3 = static_cast<int (*)(C, C)>(&nested::operator+);

  std::cout << "addresses:" << std::endl
      << "  " << func1 << "  " << func1(C{}, C{}) << std::endl
      << "  " << func2 << "  " << func2(C{}, int{}) << std::endl
      << "  " << func3 << "  " << func3(C{}, C{}) << std::endl;

  return 0;
}

/*
Output:

$ g++ -std=c++17 ./op_overload_ref.cc && ./a.out
10
addresses:
  1  10
  1  20
  1  10
*/
