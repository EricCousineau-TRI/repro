#include <iostream>

namespace nested {

struct C {
  friend int operator+(C, C) { return 10; }
};

}

using nested::C;
using nested::operator+;

typedef int (*MyFunc)(C, C);

int main() {
  std::cout << (C{} + C{}) << std::endl;

  MyFunc func1 = &operator+;

  std::cout << "addresses:" << std::endl
      << "  " << func1 << "  " << func1(C{}, C{}) << std::endl;

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
2 errors generated.
*/
