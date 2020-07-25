// From: https://github.com/boostorg/metaparse/blob/e0350c0bfe92a257bf0be2083f2a003f237b5dd8/include/boost/metaparse/v1/cpp11/string.hpp

#include <iostream>

template <char... Cs>
struct tstring {
  static constexpr int len = sizeof...(Cs);
  static constexpr char value[len + 1] = {Cs...};
};

template <typename str>
void print() {
  std::cout << "str: " << str::value << std::endl;
}

#if __has_extension(cxx_string_literal_templates)
#define TSTRING(...) tstring<__VA_ARGS__>
#else
#error "No can do :("
#endif

// template <>
// void print<tstring<"special">>() {
//   std::cout << "<specialized>" << std::endl;
// }

int main() {
  print<TSTRING("")>();
  print<TSTRING("hello world!")>();
  // print<tstring<"special">>();
  return 0;
}
