#include <cassert>
#include <cstdio>
#include <string>

#include "load_symbol.h"

int main(int argc, char** argv) {
  assert(argc == 3);
  const std::string base_b = argv[1];
  const std::string base_c = argv[2];
  func_t wrapped_b = LoadSymbol(
      Rlocation("workspace/" + base_b), "wrapped_b");
  func_t wrapped_c = LoadSymbol(
      Rlocation("workspace/" + base_c), "wrapped_c");

  const int b = (*wrapped_b)();
  const int c = (*wrapped_c)();
  printf("wrapped_b: %d\n", b);
  printf("wrapped_c: %d\n", c);
  if (b == c) {
    printf("FAIL! Separate memory for counter() -> ODR violation!\n");
    return 1;
  } else {
    printf("Success!\n");
    return 0;
  }
}
