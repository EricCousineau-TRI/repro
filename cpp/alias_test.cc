// Purpose: Test if we can forwad-declare aliases.
#include <iostream>
using namespace std;

class Alias;

void blarg(Alias* value);

class Stuff {
 public:
  void things() {
    cout << "Hello" << endl;
  }
};

using Alias = Stuff;

void blarg(Alias* value) {
  value->things();
}

int main() {
  Stuff x;
  blarg(&x);
  return 0;
}
