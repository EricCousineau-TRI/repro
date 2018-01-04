// @ref https://stackoverflow.com/questions/7863603/how-to-make-template-rvalue-reference-parameter-only-bind-to-rvalue-reference

#include <iostream>
#include <type_traits>

using namespace std;

struct Value {};

template <typename T>
struct greedy_struct {
  static void run(const T&) {
    cout << "const T& (struct)" << endl;
  }
  static void run(T&&) {
    cout << "T&& (struct)" << endl;
  }
};

// Per Toby's answer.
template <typename T>
void greedy_sfinae(T&&,
    std::enable_if_t<!std::is_rvalue_reference<T&&>::value, void*> = {}) {
  cout << "non-T&& (sfinae)" << endl;
}

template <typename T>
void greedy_sfinae(T&&,
    std::enable_if_t<std::is_rvalue_reference<T&&>::value, void*> = {}) {
  cout << "T&& (sfinae)" << endl;
}

// Bad.
template <typename T>
void greedy_sfinae_bad(T&&,
    std::enable_if_t<!std::is_rvalue_reference<T>::value, void*> = {}) {
  cout << "non-T&& (sfinae bad)" << endl;
}

template <typename T>
void greedy_sfinae_bad(T&&,
    std::enable_if_t<std::is_rvalue_reference<T>::value, void*> = {}) {
  cout << "T&& (sfinae bad)" << endl;
}

template <typename TF>
void greedy(TF&& value) {
  using T = std::decay_t<TF>;
  greedy_struct<T>::run(std::forward<TF>(value));
  greedy_sfinae(std::forward<TF>(value));
  greedy_sfinae_bad(std::forward<TF>(value));
  cout << "---" << endl;
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

const T& (struct)
non-T&& (sfinae)
non-T&& (sfinae bad)
---
const T& (struct)
non-T&& (sfinae)
non-T&& (sfinae bad)
---
T&& (struct)
T&& (sfinae)
non-T&& (sfinae bad)
---
T&& (struct)
T&& (sfinae)
non-T&& (sfinae bad)
---

Notes: Works as expected, but ugly...
*/
