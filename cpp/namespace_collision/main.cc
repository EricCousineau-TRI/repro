#include "header_1.h"
// #include "header_2.h"

// namespace alias = my_lib;
using namespace header_1;
// using namespace header_2;

namespace alias = my_lib;

int main() {
  alias::my_func();

  return 0;
}
