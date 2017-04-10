#include <iostream>

#include <drake/common/drake_path.h>

int main() {
  std::cout << drake::GetDrakePath() << std::endl;
  return 0;
}
