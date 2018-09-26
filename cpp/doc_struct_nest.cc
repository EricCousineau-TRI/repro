#include <iostream>

constexpr struct {
  struct {
    struct {
      struct {
        const char* doc = "Hello world";
        const char* doc_2 = "Hello world 2";
      } bottom;
    } mid;
  } top;
} doc;

int main() {
  std::cout << doc.top.mid.bottom.doc << std::endl;
  auto& bottom = doc.top.mid.bottom;
  std::cout << bottom.doc_2 << std::endl;
  return 0;
}
