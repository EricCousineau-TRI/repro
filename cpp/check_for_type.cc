// https://stackoverflow.com/questions/1005476/how-to-detect-whether-there-is-a-specific-member-variable-in-class
#include <iostream>
#include <utility>

#include "name_trait.h"

using std::cout;
using std::endl;

struct example_has_type {
    typedef int type;
};
NAME_TRAIT(example_has_type);

struct example_has_nothing {};
NAME_TRAIT(example_has_nothing);

struct example_has_name_func {
  void name() const {}
};
NAME_TRAIT(example_has_name_func);

// Check for ::type
// https://stackoverflow.com/a/14523787/7829525
template <typename T, typename = void>
struct has_type : std::false_type {};

template <typename T>
struct has_type<T, decltype(typename T::type{}, void())>
    : std::true_type {};

NAME_TRAIT_TPL(has_type);

// Check for .name()
template <typename T, typename = void>
struct has_name_func : std::false_type {};

// template <typename T, typename = void>
// struct has_name_func<T, decltype(std::declval<T>().name(), void())>
//     : std::true_type {};

NAME_TRAIT_TPL(has_name_func);

template <typename trait>
void print() {
  cout << name_trait<trait>::name() << "::value = " << trait::value << endl;
}

int main() {
  cout << endl;
  print<has_type<example_has_type>>();
  print<has_type<example_has_nothing>>();
  cout << endl;
  print<has_name_func<example_has_name_func>>();
  print<has_name_func<example_has_nothing>>();
  cout << endl;
  return 0;
}
