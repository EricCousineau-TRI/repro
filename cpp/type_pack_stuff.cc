#include <iostream>
#include <utility>

using namespace std;

#include "cpp/type_pack.h"
#include "cpp/name_trait.h"

template <typename... A, typename... B>
auto type_pack_concat(type_pack<A...> = {}, type_pack<B...> = {}) {
  return type_pack<A..., B...>{};
}

int main() {
  auto x = type_pack_concat(type_pack<int, void>{}, type_pack<double, void*>{});
  cout << nice_type_name<decltype(x)>() << endl;
  return 0;
}
