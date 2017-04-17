// Goal: Provide simple display on type per template binding
// Use variadics to simplify

#include <iostream>
#include <string>
using std::cout;
using std::endl;
using std::string;

template<typename T>
struct name_trait {
  static string name() { return "T"; }
};
template<typename T, typename ... Args>
struct name_trait_list {
  static string join(const string& delim = ", ") {
    return name_trait<T>::name() + delim
        + name_trait_list<Args...>::join(delim);
  }
};
template<typename T>
struct name_trait_list<T> {
  static string join(const string& delim = ", ") {
    return name_trait<T>::name();
  }
};
// // Handle empty case?
// template<>
// struct name_trait_list<> {
//   static string join(const string& delim = ", ") {
//     return "";
//   }
// };
#define NAME_TRAIT(TYPE) \
  template<> \
  struct name_trait<TYPE> { \
    static string name() { return #TYPE; } \
  };
#define NAME_TRAIT_TPL(TYPE) \
  template<typename ... Args> \
  struct name_trait<TYPE<Args...>> { \
    static string name() { \
      return #TYPE "<" + \
        name_trait_list<Args...>::join() + ">"; \
      } \
  };

// Can't handle literals...
NAME_TRAIT(int);
NAME_TRAIT(double);
NAME_TRAIT_TPL(name_trait);
NAME_TRAIT_TPL(name_trait_list);

#define PRINT(x) #x ": " << (x) << endl

int main() {
  cout
    << PRINT(name_trait<int>::name())
    << PRINT((name_trait<name_trait_list<int, double, name_trait<int>, name_trait_list<int, double>>>::name()));
  return 0;
}
