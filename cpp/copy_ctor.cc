#include <iostream>
#include <string>

using std::string;
using std::cout;
using std::endl;

class Test {
public:
  Test(const string& name = string("default"))
    : name_(name)
  { }

  const string& name() const {
    return name_;
  }

private:
  string name_;
};

int main() {
  {
    Test test;
    cout << test.name() << endl;
  }
  {
    Test test("nondefault");
    cout << test.name() << endl;
  }
  return 0;
}
