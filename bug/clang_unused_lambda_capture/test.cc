#include <iostream>

constexpr struct {
  struct {
    const char* doc = "Value";
  } mid;
} top;

int main() {
  auto& mid = top.mid;
  auto bind = [&mid](auto) {
    std::cout << "Value: " << mid.doc << std::endl;
  };
  bind(int{});
  return 0;
}
