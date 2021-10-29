#include <iostream>

#include "name_trait.h"

using std::cout;

int x{};

const int& ret_cref() { return x; }

int main() {
  // For lambdas.
  auto lambda_simple = []() { return ret_cref(); };
  cout << PRINT(name_trait<decltype(lambda_simple())>::name());
  auto lambda_auto = []() -> auto { return ret_cref(); };
  cout << PRINT(name_trait<decltype(lambda_auto())>::name());
  auto lambda_auto_rref = []() -> auto&& { return ret_cref(); };
  cout << PRINT(name_trait<decltype(lambda_auto_rref())>::name());
  auto lambda_auto_decl = []() -> decltype(auto) { return ret_cref(); };  
  cout << PRINT(name_trait<decltype(lambda_auto_decl())>::name());
}

/**
Output:

>>> name_trait<decltype(lambda_simple())>::name()
int

>>> name_trait<decltype(lambda_auto())>::name()
int

>>> name_trait<decltype(lambda_auto_rref())>::name()
const int&

>>> name_trait<decltype(lambda_auto_decl())>::name()
const int&
*/