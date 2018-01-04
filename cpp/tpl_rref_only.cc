// @ref https://stackoverflow.com/questions/7863603/how-to-make-template-rvalue-reference-parameter-only-bind-to-rvalue-reference

#include <iostream>
#include <type_traits>

using namespace std;

struct Value {};

template <typename T>
void greedy(const T&) {
  cout << "const T&" << endl;
}

template <typename T, typename =
    std::enable_if_t<std::is_rvalue_reference<T>::value>>
void greedy(T&&) {
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

const T&
const T&
const T&
const T&

Notes: Why the last two???
*/
