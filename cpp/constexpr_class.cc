#include <iostream>
#include <string>

using namespace std;

#define EVAL(x) std::cout << ">>> " #x ";" << std::endl; x; std::cout << std::endl
#define PRINT(x) ">>> " #x << std::endl << (x) << std::endl << std::endl

class TestClass {
 public:
  constexpr TestClass(int value, const char* name)
    : value_(value), name_(name) {}

  int value() const { return value_; }
  string name() const { return name_; }

  void set_value(int value) { value_ = value; }
  void set_name(const char* name) { name_ = name; }
 private:
  int value_{};
  const char* name_{};
};

class Bleh {
 public:
  Bleh(int value)
    : value_(value) {}
  int value() const { return value_; }
 private:
  int value_{};
};

// TODO(eric.cousineau): Replace with the success of is_literal_type.
// @ref 

int main() {
  cout
    << PRINT(is_literal_type<TestClass>::value)
    << PRINT(is_literal_type<Bleh>::value);

  // constexpr Bleh a(5);  // Compiler error.
  constexpr TestClass b(1, "Hello");

  cout
    << PRINT(b.value())
    << PRINT(b.name());

  TestClass c(2, "World");
  c.set_value(10);
  c.set_name("World!!!");
  cout
    << PRINT(c.value())
    << PRINT(c.name());

  return 0;
}
