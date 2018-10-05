#include <iostream>

struct Top {
  struct Mid {
    const char* doc = "Value";
  };
  Mid mid;
};

// If this were local, no warning would be produced.
Top top;

int main() {
  auto& mid = top.mid;
  auto bind = [&mid](auto) {
    std::cout << "Value: " << mid.doc << std::endl;
  };
  bind(int{});
  return 0;
}
