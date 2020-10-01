#include <iostream>

namespace nested {

struct C {
  friend int operator+(C, C) { return 10; }
  friend int operator+(C, int) { return 20; }
};

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

$ clang++-9 -std=c++17 ./op_overload_ref.cc && ./a.out
./op_overload_ref.cc:13:15: error: no member named 'operator+' in namespace 'nested'
using nested::operator+;
      ~~~~~~~~^
./op_overload_ref.cc:20:19: error: use of undeclared 'operator+'
  MyFunc func1 = &operator+;
                  ^
./op_overload_ref.cc:21:54: error: no member named 'operator+' in namespace 'nested'
  auto func2 = static_cast<int (*)(C, int)>(&nested::operator+);
                                             ~~~~~~~~^
./op_overload_ref.cc:22:52: error: no member named 'operator+' in namespace 'nested'
  auto func3 = static_cast<int (*)(C, C)>(&nested::operator+);
                                           ~~~~~~~~^
4 errors generated.
*/
