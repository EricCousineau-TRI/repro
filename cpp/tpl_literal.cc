#include <iostream>

#include "cpp/name_trait.h"

using namespace std;

// template <typename T, T Value>
// struct literal_tag_impl {
//   using type = T;
//   static constexpr T value = Value;
// };

// template <int Value>
// using literal_tag = std::integral_constant<int, Value>;

// template <size_t Value>
// using literal_tag = std::integral_constant<size_t, Value>;

// template <typename T>
// constexpr auto literal_tag_impl(T value) {
//   return value;
// }

template <typename T>
constexpr auto literal_tag(T value) {
  // constexpr auto c_value = value;
  return std::integral_constant<T, value>{};
}

int main() {
  cout << literal_tag(10) << endl;
  // cout << nice_type_name<literal_tag<2>>() << endl;
  return 0;
}
