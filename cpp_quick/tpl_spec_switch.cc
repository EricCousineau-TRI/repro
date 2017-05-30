#include "tpl_spec_switch.h"

// Test against linker errors / multiple definitions.
int extra_stuff() {
  Example<int> ex_int;
  Example<A> ex_a;

  return ex_int.Stuff() + ex_a.Stuff();
}
