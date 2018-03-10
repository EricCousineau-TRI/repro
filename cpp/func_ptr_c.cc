#include <iostream>
using namespace std;

using Func = void(*)(int);

Func make_func() {
  Func out = [](int x) {
    cout << "Valid func " << x << endl;
  };
  cout << "Return" << endl;
  return out;
}

int main() {
  Func f = make_func();
  f(10);
  return 0;
}
