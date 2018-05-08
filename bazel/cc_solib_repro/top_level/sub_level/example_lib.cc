#include "example_lib.h"

int upstream();

int func() {
  return upstream() / 2;
}
