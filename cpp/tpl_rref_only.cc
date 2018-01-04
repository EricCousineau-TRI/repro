// @ref https://stackoverflow.com/questions/7863603/how-to-make-template-rvalue-reference-parameter-only-bind-to-rvalue-reference

#include <iostream>
#include <type_traits>

using namespace std;

struct Value {};

template <typename T>
using direct = T; //std::conditional_t<std::is_same<T, T>::value, T, void>;

template <typename T>
using rvalue = std::add_rvalue_reference_t<T>; //direct<T>&&;

template <typename T>
void greedy(const T&) {
  cout << "const T&" << endl;
}

template <typename T>
void greedy(direct<T&&>) {
  cout << "T&&" << endl;
}

int main() {
  Value x;
  const Value y;

  greedy(x);
  greedy(y);
  greedy(Value{});
  greedy(std::move(x));

  return 0;
}

/*
Output:

T&&
const T&
T&&
T&&

Notes: Still too greedy?
*/
