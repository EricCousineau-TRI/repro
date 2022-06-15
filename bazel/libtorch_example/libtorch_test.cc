#include <iostream>

#include <torch/torch.h>

int main() {
  torch::Tensor tensor = torch::rand({5, 10});
  tensor = tensor.cuda();
  std::cout << "Example tensor:\n" << tensor << std::endl;
  return 0;
}
