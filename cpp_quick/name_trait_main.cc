// Goal: Provide simple display on type per template binding
// Use variadics to simplify
#include "name_trait.h"

#include <iostream>
using std::cout;
using std::endl;
using std::string;

int main() {
  cout
    << PRINT(name_trait<int>::name())
    << PRINT((name_trait<name_trait_list<int, double, name_trait<int>, name_trait_list<int, double>>>::name()))
    // For unknown, should print "T"
    << PRINT(name_trait<std::iostream>::name())
    << PRINT(name_trait<name_trait<std::iostream>>::name())
    << PRINT(name_trait<int&>::name())
    << PRINT(name_trait<const double&>::name())
    << PRINT(name_trait<string&&>::name())
    << PRINT(name_trait<int*>::name())
    << PRINT(name_trait<const char*>::name())
    << PRINT(name_trait<char * const&>::name())
    << PRINT(name_trait<decltype("Hello")>::name())
    << PRINT(name_trait<decltype("Hello"[0])>::name())
    << PRINT(name_trait<int[]>::name());
  return 0;
}
