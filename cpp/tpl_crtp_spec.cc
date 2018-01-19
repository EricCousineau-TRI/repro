#include <iostream>
#include <type_traits>

// #include "cpp/name_trait.h"
#define PRINT(x) ">>> " #x << std::endl << (x) << std::endl << std::endl


using namespace std;

template <template <typename> class Tpl>
struct is_base_tpl_of_impl {
  template <typename Base>
  static std::true_type check(const Tpl<Base>&);
  static std::false_type check(...);

  template <typename Derived>
  using result = decltype(check(std::declval<Derived>()));
};

template <template <typename> class Tpl, typename Derived>
using is_base_tpl_of =
    typename is_base_tpl_of_impl<Tpl>::template result<Derived>;

template <typename Base>
struct alias_wrapper : public Base {};

// template <typename Alias, typename = void>
// struct alias_wrapper_of {
//     using type = alias_wrapper<Alias>;
// };

// template <>
// struct alias_wrapper_of<void, void> {
//     using type = void;
// };

// template <typename Alias>
// struct alias_wrapper_of<alias_wrapper<Alias>> {
//     using type = alias_wrapper<Alias>;
// };


struct A {};
struct B : public A {};
using C = alias_wrapper<A>;
struct D : public C {};

int main() {
  cout
    << PRINT((is_base_tpl_of<alias_wrapper, A>::value))
    << PRINT((is_base_tpl_of<alias_wrapper, B>::value))
    << PRINT((is_base_tpl_of<alias_wrapper, C>::value))
    << PRINT((is_base_tpl_of<alias_wrapper, D>::value));
}
