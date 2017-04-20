#pragma once

#include <string>
#include <utility>
#include <sstream>

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
template<typename T>
struct name_trait<T&> {
  static std::string name() { return name_trait<T>::name() + "&"; }
};
template<typename T>
struct name_trait<T&&> {
  static std::string name() { return name_trait<T>::name() + "&&"; }
};
template<typename T>
struct name_trait<const T> {
  static std::string name() { return "const " + name_trait<T>::name(); }
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

// Can't automatically handle literals as template parameters...
NAME_TRAIT(bool);
NAME_TRAIT(char);
NAME_TRAIT(int);
NAME_TRAIT(double);
NAME_TRAIT(std::string);
NAME_TRAIT_TPL(name_trait);
NAME_TRAIT_TPL(name_trait_list);

// Specialize name_trait for std::size_t, and use different
template<std::size_t T>
struct literal_trait {
  static std::string name() {
    std::ostringstream os;
    os << T;
    return os.str();
  }
};
template<std::size_t T, std::size_t ... Args>
struct literal_trait_list {
  static std::string join(const std::string& delim = ", ") {
    return literal_trait<T>::name() + delim
        + literal_trait_list<Args...>::join(delim);
  }
};
template<std::size_t T>
struct literal_trait_list<T> {
  static std::string join(const std::string& delim = ", ") {
    return literal_trait<T>::name();
  }
};

#define NAME_TRAIT_TPL_LITERAL(TYPE) \
  template<std::size_t ... Args> \
  struct name_trait<TYPE<Args...>> { \
    static std::string name() { \
      return #TYPE "<" + \
        literal_trait_list<Args...>::join() + ">"; \
      } \
  };
NAME_TRAIT_TPL_LITERAL(std::index_sequence);

// Random helper macros
#define PRINT(x) ">>> " #x << std::endl << (x) << std::endl
