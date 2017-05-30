#include <iostream>

#include "tpl_spec_switch.h"

using std::cout;
using std::endl;

#define EVAL(x) std::cout << ">>> " #x ";" << std::endl; x; cout << std::endl
#define PRINT(x) ">>> " #x << std::endl << (x) << std::endl << std::endl

class B : public A {
 public:
  static constexpr int value = 100;
};

int main() {
  Example<int> ex_int;
  Example<A> ex_a;
  Example<B> ex_b;

  cout
      << PRINT(ex_int.Stuff())
      << PRINT(ex_a.Stuff())
      << PRINT(ex_b.Stuff());

  extra_stuff();

  return 0;
}
