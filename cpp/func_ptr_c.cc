#include <iostream>
using namespace std;

using Func = void(*)(int);

// Func make_func() {
//   Func out = [](int x) {
//     cout << "Valid func " << x << endl;
//   };
//   cout << "Return" << endl;
//   return out;
// }

template <int* x>
void shibble() {
  *x += 1;
}

template <const Func& f>
void call() {
  f(10);
}

// void yar(int x) {}



int main() {
  // Func f = make_func();
  // Func f2 = [](int x) { cout << "Yar " << x << endl; };
  // call<&yar>();

  // f(10);

  static int x = 1;
  shibble<&x>();
  cout << x << endl;

  struct Biscuit {
    static void shit(int) {
      cout << "Bullshit\n";
    }
  };
  static const Func a = [] (int y) {
    cout << "Less bullshit\n";
  };
  Func b = [](int x) {
    cout << "Moar bullshit\n";
    Biscuit::shit(x);
    a(x);
    // call<a>();
  };

  b(1);

  return 0;
}
