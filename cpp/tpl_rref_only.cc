// @ref https://stackoverflow.com/questions/7863603/how-to-make-template-rvalue-reference-parameter-only-bind-to-rvalue-reference

#include <iostream>
#include <type_traits>

using namespace std;

struct Value {};

template <typename T>
using rvalue = T&&;

template <typename T>
void greedy(const T&) {
  cout << "const T&" << endl;
}

template <typename T>
void greedy(rvalue<T>) {
  cout << "T&&" << endl;
}

int main() {
  Value x;
  const Value y;

  greedy(x);
  greedy(y);
  greedy(Value{});
  greedy(std::move(Value{}));

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
