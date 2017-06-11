#include <iostream>
#include <string>

using namespace std;

struct A {
  int value{10};
  string name{"Howdy"};
};

ostream& operator<<(ostream& os, const A& a) {
  os
    << "value: " << a.value << endl
    << "name: " << a.name << endl
    << "---" << endl;
  return os;
}

int main() {
  A a {.value = 22};
  A b;
  A c {25, "Hola"};
  auto d = A {.name = "You",};

  cout << a << b << c << d << endl;
}
