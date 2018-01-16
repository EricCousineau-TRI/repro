#include <iostream>

using namespace std;

template <template <typename...> class Tpl>
struct template_tag {
  template <typename ... Ts>
  using bind = Tpl<Ts...>;
};

template <template <typename> class Tpl>
struct template_single_tag {
  template <typename T>
  using bind = Tpl<T>;
};

// // Cannot do this: Mixes parameter types.
// template <template <typename...> class Tpl>
// struct template_single_tag<Tpl, void> : public template_tag<Tpl> {};

template <typename T>
struct single_tpl {};

template <typename T>
using single_tpl_alias = single_tpl<T>;

template <typename ... Ts>
using single_tpl_alias_pack = single_tpl<Ts...>;

template <typename T, typename U>
struct double_tpl {};

template <typename T, typename U>
using double_tpl_alias = double_tpl<T, U>;

template <typename T, typename ... Ts>
struct pack_tpl {};

template <typename T, typename ... Ts>
using pack_tpl_alias = pack_tpl<T, Ts...>;

template <template <typename> class Tpl>
auto create_template_tag(template_single_tag<Tpl> tag = {}) {
  return tag;
}

// // Ambiguous with above.
// template <template <typename...> class Tpl>
// auto create_template_tag(template_tag<Tpl> tag = {}) {
//   return tag;
// }

int main() {
  {
    template_single_tag<single_tpl> tag{};
    decltype(tag)::bind<int> t{};
    (void)t;
  }

  {
    template_tag<single_tpl> tag{};
    decltype(tag)::bind<int> t{};
    (void)t;
  }

  // Try aliasing.
  {
    // template_tag<single_tpl_alias> tag{};  // Does NOT work. 
    // auto tag = create_template_tag<nested::type>();  // Cannot overload.
    template_single_tag<single_tpl_alias> tag{};
    decltype(tag)::bind<int> t{};
    (void)t;
  }
  {
    template_tag<single_tpl_alias_pack> tag{};  // DOES work.
    decltype(tag)::bind<int> t{};
    (void)t;
  }

  {
    template_tag<double_tpl> tag{};
    decltype(tag)::bind<int, double> t{};
    (void)t;
  }

  // {
  //   // Does NOT work.
  //   template_tag<double_tpl_alias> tag{};
  //   decltype(tag)::bind<int, double> t{};
  //   (void)t;
  // }

  {
    template_tag<pack_tpl> tag{};
    decltype(tag)::bind<int, double, char> t{};
    (void)t;
  }

  // {
  //   // Does NOT work.
  //   template_tag<pack_tpl_alias> tag{};
  //   decltype(tag)::bind<int, double, char> t{};
  //   (void)t;
  // }

  return 0;
}
