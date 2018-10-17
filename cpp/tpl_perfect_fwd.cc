#include <iostream>

#include "cpp/name_trait.h"

#define R_EVAL(x) \
    cout << nice_type_name<decltype(x)>() << " = Return" << endl; \
    EVAL(x)

using namespace std;

template <typename T>
T&& func(T&& value) {
  cout
    << nice_type_name<T>() << " = T" << endl
    << nice_type_name<T&&>() << " = T&&" << endl
    << nice_type_name<decltype(value)>() << " = decltype(value)" << endl;
  return std::forward<T>(value);
}

struct Value {};

int main() {
  const Value x_const;
  Value x;

  R_EVAL(func(x_const));
  R_EVAL(func(x));
  R_EVAL(func(std::move(x)));
  R_EVAL(func(std::move(x_const)));

  return 0;
}

/*
Output:

const Value& = Return
>>> func(x_const);
const Value& = T
const Value& = T&&
const Value& = decltype(value)

Value& = Return
>>> func(x);
Value& = T
Value& = T&&
Value& = decltype(value)

Value&& = Return
>>> func(std::move(x));
Value = T
Value&& = T&&
Value&& = decltype(value)

const Value&& = Return
>>> func(std::move(x_const));
const Value = T
const Value&& = T&&
const Value&& = decltype(value)

*/
