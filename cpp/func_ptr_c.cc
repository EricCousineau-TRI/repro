#include <iostream>
using namespace std;

using Func = void(*)();

template <Func a>
Func wrap() {
  return []() { a(); };
}
// Cannot be in function scope.
struct tmp { static void a2() { cout << "A2\n"; } };

template <typename Arg, void (*func)(Arg)>
void call(Arg arg) {
  cout << "Wrap\n";
  func(arg);
}

template <typename ArgT>
struct signature {
  using Arg = ArgT;
};

template <typename Arg>
signature<Arg> infer(void (*)(Arg));

template <typename F, F f, typename T>
void infer(T x) {
  using Sig = decltype(infer(f));
  call<typename Sig::Arg, f>(x);
}

void stuff(int x) {
  cout << "stuff(" << x << ")\n";
}

int main() {
  {
    Func b = wrap<tmp::a2>();
    b();
  }

  {
    static const Func a = []() { cout << "A3\n"; };
    Func b = []() { a(); };
    b();
  }

  {
    call<int, stuff>(10);
    infer<decltype(stuff), stuff>(10);
  }

  // {
  //   auto a = []() { cout << "A3\n"; };
  //   using A = decltype(a);
  //   Func b = []() { A a{}; a(); };
  //   b();
  // }

  return 0;
}
