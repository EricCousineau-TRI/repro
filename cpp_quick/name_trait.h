#pragma once

#include <string>


template<typename T>
struct name_trait {
  static std::string name() { return "T"; }
};
template<typename T, typename ... Args>
struct name_trait_list {
  static std::string join(const std::string& delim = ", ") {
    return name_trait<T>::name() + delim
        + name_trait_list<Args...>::join(delim);
  }
};
template<typename T>
struct name_trait_list<T> {
  static std::string join(const std::string& delim = ", ") {
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
    static std::string name() { return #TYPE; } \
  };
#define NAME_TRAIT_TPL(TYPE) \
  template<typename ... Args> \
  struct name_trait<TYPE<Args...>> { \
    static std::string name() { \
      return #TYPE "<" + \
        name_trait_list<Args...>::join() + ">"; \
      } \
  };

// Can't handle literals as template parameters...
NAME_TRAIT(bool);
NAME_TRAIT(char);
NAME_TRAIT(int);
NAME_TRAIT(double);
NAME_TRAIT_TPL(name_trait);
NAME_TRAIT_TPL(name_trait_list);

#define PRINT(x) ">>> " #x << endl << (x) << std::endl
