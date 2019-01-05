// Purpose: Make a functor-ish type thing that optionally handles scalar
// conversion.

#include <iostream>
#include <functional>
#include <vector>

// Fixed number of types.
using A = int;
using B = float;
using C = char;

/// Holds a set of functors for specific scalar types.
template <template <typename> class Signature>
struct ScalarFunctorSet {
  Signature<A> a;
  Signature<B> b;
  Signature<C> c;

  ScalarFunctorSet(Signature<A> a_in, Signature<B> b_in, Signature<C> c_in)
    : a(a_in), b(b_in), c(c_in) {}

  template <typename Lambda>
  ScalarFunctorSet(Lambda f)
    : a(f), b(f), c(f) {}
};

// Specific example.
const char* get_name(int) { return "int"; }
const char* get_name(float) { return "float"; }
const char* get_name(char) { return "char"; }

template <typename T>
using MyFunc = std::function<void (const T&, const std::vector<T>&)>;

void simple_func(int x, std::vector<int> y) {
  std::cout << "simple_func(" << x << ", vec[" << y.size() << "])" << std::endl;
}

template <typename T>
void generic_func(T x, std::vector<T> y) {
  std::cout << "generic_func<" << get_name(x) << ">(" << x << ", vec[" << y.size() << "])" << std::endl;
}

#define EVAL(x) std::cout << ">>> " #x ";" << std::endl; x; std::cout << std::endl

int main() {
  // Int only.
  ScalarFunctorSet<MyFunc> only_int(&simple_func, {}, {});
  // All (generic).
  ScalarFunctorSet<MyFunc> all(
      [](auto x, auto y){ return generic_func(x, y); });

  EVAL(only_int.a(1, {2}));
  EVAL(all.a(1, {3, 4}));
  EVAL(all.b(1.5, {5., 6., 7.}));
  EVAL(all.c('z', {'a', 'b', 'c', 'd'}));
  return 0;
}

/*
Output:

>>> only_int.a(1, {2});
simple_func(1, vec[1])

>>> all.a(1, {3, 4});
generic_func<int>(1, vec[2])

>>> all.b(1.5, {5., 6., 7.});
generic_func<float>(1.5, vec[3])

>>> all.c('z', {'a', 'b', 'c', 'd'});
generic_func<char>(z, vec[4])

*/
