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

struct B {
  A a {.value = 50};  // NOTE: Any partial construction will use A's original default ctor.
  int extra {22};
};

ostream& operator<<(ostream& os, const B& b) {
  os
    << "a.value: " << b.a.value << endl
    << "a.name: " << b.a.name << endl
    << "extra: " << b.extra << endl
    << "---" << endl;
  return os;
}

int main() {
  A a {.value = 22};
  A b;
  A c {25, .name = "Hola", };
  // A c {25, .name = "Hola", .value = 200};  // Will generate warning.
  auto d = A {.name = "You", .value = 15,};

  cout << a << b << c << d << endl;

  B e { .a = {.name = "Whadup"}, .extra = -100};
  cout << e << endl;
}
