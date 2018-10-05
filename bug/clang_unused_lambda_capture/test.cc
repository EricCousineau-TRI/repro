#include <iostream>

// Only issues warning for non-automatic storage?
const char* top_doc = "Value";

int main() {
  auto& doc = top_doc;
  [&doc]() {
    std::cout << "Value: " << doc << std::endl;
  }();
  return 0;
}
