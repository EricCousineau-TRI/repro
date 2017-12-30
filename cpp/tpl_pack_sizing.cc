#include <array>
#include <iostream>
#include <utility>

using namespace std;

template <typename T, typename ... Ts, int N = 1 + sizeof...(Ts)>
array<T, N> do_stuff(const T& arg, const Ts&... args) {
  return array<T, N>{{arg, args...}};
}

// // Cannot implicitly expand index pack.
// template <size_t ... Is>
// void check(
//     std::index_sequence<Is...> = std::make_index_sequence<5>{}) {  // = std::make_index_sequence<sizeof...(Ts)>{}
//   cout << "Whoo" << endl;
// }

int main() {
  auto x = do_stuff(1, 2, 3);
  // check<>();
  struct check {
    // template <typename T>
    // void run() { }
  };

  check stuff;
  return 0;
}
