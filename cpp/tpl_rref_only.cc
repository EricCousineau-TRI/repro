// @ref https://stackoverflow.com/questions/7863603/how-to-make-template-rvalue-reference-parameter-only-bind-to-rvalue-reference

#include <iostream>
#include <type_traits>

#include "cpp/name_trait.h"

using namespace std;

struct Value {};

// template <typename T>
// using direct = T; //std::conditional_t<std::is_same<T, T>::value, T, void>;

// template <typename T>
// using rvalue = std::add_rvalue_reference_t<T>; //direct<T>&&;

// template <typename T>
// using is_const_lvalue_reference =
//     std::integral_constant<bool,
//         std::is_const<T>::value && std::is_lvalue_reference<T>::value>;

template <typename T>
struct greedy_impl {
  static void run(const T&) {
    cout << "const T&: " << nice_type_name<T>() << endl;
  }
  static void run(T&&) {
    cout << "T&&: " << nice_type_name<T>() << endl;
  }
};

template <typename TF>
void greedy(TF&& value) {
  using T = std::decay_t<TF>;
  greedy_impl<T>::run(std::forward<TF>(value));
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

const T&: Value
const T&: Value
T&&: Value
T&&: Value

Notes: Works as expected, but ugly...
*/
