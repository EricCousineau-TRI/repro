#include <iostream>
#include <string>

using namespace std;

const string value =
    R"_(.
Hello
world
.)_";

int main() {
  cout << "---\n";
  cout << value;
  cout << "---\n";
  return 0;
}
