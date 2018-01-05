#include <iostream>

#include "cpp/name_trait.h"

using namespace std;

template <typename T>
void func(T&& value) {
  cout
    << nice_type_name<T>() << " = T" << endl
    << nice_type_name<T&&>() << " = T&&" << endl
    << nice_type_name<decltype(value)>() << " = decltype(value)" << endl;
}

struct Value {};

int main() {
  const Value x_const;
  Value x;

  EVAL(func(x_const));
  EVAL(func(x));
  EVAL(func(std::move(x)));

  return 0;
}

/*
Output:

>>> func(x_const);
const Value& = T
const Value& = T&&
const Value& = decltype(value)

>>> func(x);
Value& = T
Value& = T&&
Value& = decltype(value)

>>> func(std::move(x));
Value = T
Value&& = T&&
Value&& = decltype(value)

*/
