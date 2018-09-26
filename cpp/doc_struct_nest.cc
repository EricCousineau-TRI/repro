#include <iostream>

#include "doc_struct_nest_big.h"

using str = const char*;

constexpr struct /* doc */ {

  struct /* top */ {

    struct /* mid */ {

      struct /* bottom */ {
        const char* doc[2] = {
          "Hello world",
          "Hello world 2",
        };

      } bottom;

    } mid;

  } top;

} doc;

int main() {
  std::cout << doc.top.mid.bottom.doc[0] << std::endl;
  auto& bottom = doc.top.mid.bottom;
  std::cout << bottom.doc[1] << std::endl;
  auto& cls = root.drake.multibody.MultibodyForces;
  std::cout << cls.num_bodies.doc << std::endl;
  return 0;
}
