#include <iostream>

using namespace std;

template <template <typename> class Tpl>
struct template_tag {
  template <typename T>
  using bind = Tpl<T>;
};

template <template <typename...> class Tpl>
struct template_pack_tag {
  template <typename ... Ts>
  using bind = Tpl<Ts...>;
};


template <typename T>
struct single_tpl {};

template <typename T, typename U>
struct double_tpl {};

template <typename T, typename ... Ts>
struct pack_tpl {};

struct nested {
  template <typename T>
  using type = single_tpl<T>;
};

int main() {
  {
    template_tag<single_tpl> tag{};
    decltype(tag)::bind<int> t{};
    (void)t;
  }

  {
    template_pack_tag<single_tpl> tag{};
    decltype(tag)::bind<int> t{};
    (void)t;
  }

  {
    template_pack_tag<double_tpl> tag{};
    decltype(tag)::bind<int, double> t{};
    (void)t;
  }

  {
    template_pack_tag<pack_tpl> tag{};
    decltype(tag)::bind<int, double, char> t{};
    (void)t;
  }

  // Try nesting.
  {
    // template_pack_tag<nested::type> tag{};  // Does not work???
    template_tag<nested::type> tag{};
    decltype(tag)::bind<int> t{};
    (void)t;
  }

  return 0;
}
