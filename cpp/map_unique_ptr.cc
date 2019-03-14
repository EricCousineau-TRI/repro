#include <iostream>
#include <map>
#include <memory>

using IntMap = std::map<int, std::unique_ptr<int>>;

void func(IntMap value) {
  for (auto& pair : value) {
    std::cout << pair.first << " " << pair.second << std::endl;
  }
}

int main() {
  IntMap value;
  for (int i : {0, 1, 2, 3}) {
    value.push_back({i, std::make_unique<int>(i * 10)});
  }
  func(std::move(value));
  return 0;
}
