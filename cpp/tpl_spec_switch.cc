#include "tpl_spec_switch.h"

// Test against linker errors / multiple definitions.
int extra_stuff() {
  Example<A> ex_a;
  Example<B> ex_b;

  return ex_a.Stuff() + ex_b.Stuff();
}
