#include <iostream>
#include <utility>

using namespace std;

#include "cpp/type_pack.h"
#include "cpp/name_trait.h"

template <typename T>
using identity_t = T;
template <typename T>
using ptr_t = T*;

template <typename... A, typename... B>
auto type_pack_concat(type_pack<A...> = {}, type_pack<B...> = {}) {
  return type_pack<A..., B...>{};
}

template <template <typename> class Apply, typename... T>
auto type_pack_apply(type_pack<T...> = {}) {
  return type_pack<Apply<T>...>{};
}

int main() {
  using A = type_pack<int, void>;
  using B = type_pack<double, void*>;
  auto x = type_pack_concat(A{}, B{});
  cout << nice_type_name<decltype(x)>() << endl;
  auto y = type_pack_apply<ptr_t>(x);
  cout << nice_type_name<decltype(y)>() << endl;
  return 0;
}
