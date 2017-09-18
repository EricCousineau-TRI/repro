#include <iostream>
#include <string>

using namespace std;

const string value = R"(.
Hello
world
.)";

int main() {
  cout << "---\n";
  cout << value;
  cout << "---\n";
  return 0;
}
