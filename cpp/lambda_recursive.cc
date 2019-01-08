#include <iostream>

int main() {
  // No dice.
  auto recurse = [&](int n) -> int {
    if (n == 0) return 0;
    else return recurse(n - 1);
  };
  std::cout << recurse(10) << std::endl;
  return 0;
}
