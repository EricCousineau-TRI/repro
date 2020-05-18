/*
Purpose: Try to figure out what compilers actually support this?
https://en.cppreference.com/w/cpp/language/extending_std#Adding_template_specializations

Answer: https://stackoverflow.com/a/25594741/7829525
*/

#include <functional>
#include <iostream>

struct MyType {};

template <>
struct std::hash<MyType> {
  std::size_t operator()(const MyType&) { return 42; }
};

int main() {
  std::cout << "hash<MyType>: " << std::hash<MyType>{}(MyType{}) << std::endl;
  return 0;
}
