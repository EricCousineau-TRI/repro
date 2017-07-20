#include <experimental/optional>
#include <iostream>

using std::cout;
using std::endl;
using std::experimental::optional;

class Test {
  Test(int value)
      : value_(value) {}
  int value() const { return value_; }
 public:
  int value_;
};

int main() {
  optional<Test> value;
  cout << "Good to go" << endl;
  return 0;
}
