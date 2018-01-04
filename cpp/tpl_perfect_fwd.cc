#include <iostream>

#include "cpp/name_trait.h"

using namespace std;

template <typename T>
void func(T&& value) {
  cout
    << nice_type_name<T>() << " = T" << endl
    << nice_type_name<decltype(value)>() << " = decltype(value)" << endl;
}

struct Value {};

int main() {
  cout << R"""(
template <typename T>
void func(T&& value) {
  cout
    << nice_type_name<T>() << " = T" << endl
    << nice_type_name<decltype(value)>() << " = decltype(value)" << endl;
}

)""";
  const Value x_const;
  Value x;

  cout << nice_type_name<const int&>() << endl;

  EVAL(func(x_const));
  EVAL(func(x));
  EVAL(func(std::move(x)));

  return 0;
}

/*
Output:

template <typename T>
void func(T&& value) {
  cout
    << nice_type_name<T>() << " = T" << endl
    << nice_type_name<decltype(value)>() << " = decltype(value)" << endl;
}

const int&
>>> func(x_const);
const Value& = T
const Value& = decltype(value)

>>> func(x);
Value& = T
Value& = decltype(value)

>>> func(std::move(x));
Value = T
Value&& = decltype(value)
*/
