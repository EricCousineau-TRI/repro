#include <iostream>

struct MyStruct {
  std::byte stuff[10] = {};
  int my_field{};
};

int main() {
  using MemberPointer = int MyStruct::*;
  MemberPointer input = &MyStruct::my_field;

  // bad mojo
  void* expected_offset = reinterpret_cast<void*>(10);
  auto output = reinterpret_cast<MemberPointer&>(expected_offset);

  std::cout << (input == output) << std::endl;

  return 0;
}
