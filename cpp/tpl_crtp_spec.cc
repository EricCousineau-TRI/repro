#include <iostream>
#include <type_traits>

#include "cpp/name_trait.h"

using namespace std;

template <template <typename...> class Tpl>
struct is_base_template_of_impl {
  template <typename ... Base>
  static std::true_type check(const Tpl<Base...>*);
  static std::false_type check(void*);
};

template <template <typename...> class Tpl, typename Derived>
using is_base_template_of =
    decltype(is_base_template_of_impl<Tpl>::check((Derived*){}));

template <typename Base>
struct alias_wrapper : public Base {};


template <typename Alias, bool already_wrapped = false>
struct alias_wrapper_of_impl {
  using type = alias_wrapper<Alias>;
};

template <typename Alias>
struct alias_wrapper_of_impl<Alias, true> {
  using type = Alias;
};

template <bool already_wrapped>
struct alias_wrapper_of_impl<void, already_wrapped> {
  using type = void;
};

template <typename Alias>
using alias_wrapper_of =
    typename alias_wrapper_of_impl<
        Alias, is_base_template_of<alias_wrapper, Alias>::value>::type;

struct A {};
struct B : public A {};
using C = alias_wrapper<A>;
struct D : public C {};

using Aw = alias_wrapper_of<A>;
using Bw = alias_wrapper_of<B>;
using Cw = alias_wrapper_of<C>;
using Dw = alias_wrapper_of<D>;

int main() {
  cout
    << PRINT((is_base_template_of<alias_wrapper, A>::value))
    << PRINT((is_same<Aw, alias_wrapper<A>>::value))
    << nice_type_name<A>() << " -> " << nice_type_name<Aw>() << endl
    << PRINT((is_base_template_of<alias_wrapper, B>::value))
    << PRINT((is_same<Bw, alias_wrapper<Bw>>::value))
    << nice_type_name<B>() << " -> " << nice_type_name<Bw>() << endl
    << PRINT((is_base_template_of<alias_wrapper, C>::value))
    << PRINT((is_same<Cw, C>::value))
    << nice_type_name<C>() << " -> " << nice_type_name<Cw>() << endl
    << PRINT((is_base_template_of<alias_wrapper, D>::value))
    << PRINT((is_same<Dw, D>::value))
    << nice_type_name<D>() << " -> " << nice_type_name<Dw>() << endl
    << PRINT((is_base_template_of<alias_wrapper, void>::value));
}
