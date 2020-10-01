#include <iostream>

namespace nested {

class C {
 private:
  friend int operator+(C, C) { return C::kSecret; }
  static constexpr int kSecret = 100;
};

// int operator+(C, C);

}

using nested::C;
// using nested::operator+;

typedef int (*MyFunc)(C, C);

int main() {
  std::cout << (C{} + C{}) << std::endl;

  MyFunc func1 = &nested::C::operator+;

  std::cout
      << "&func1=" << func1 << ",  value=" << func1(C{}, C{}) << std::endl;

  return 0;
}

/*
Output:

$ clang++-9 -std=c++17 ./op_overload_ref.cc && ./a.out
./op_overload_ref.cc:23:30: error: no member named 'operator+' in 'nested::C'
  MyFunc func1 = &nested::C::operator+;
                  ~~~~~~~~~~~^
1 error generated.

*/
