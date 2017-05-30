#include <iostream>

#include "tpl_spec_switch.h"

using std::cout;
using std::endl;

#define PRINT(x) ">>> " #x << std::endl << (x) << std::endl << std::endl

class C : public A {
 public:
  static constexpr int value = 100;
};

int main() {
  Example<A> ex_a;
  Example<B> ex_b;
  Example<C> ex_c;

  cout
      << PRINT(ex_a.Stuff())
      << PRINT(ex_b.Stuff())
      << PRINT(ex_c.Stuff())
      << PRINT(extra_stuff());

  return 0;
}
