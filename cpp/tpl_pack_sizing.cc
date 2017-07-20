#include <array>
#include <iostream>

using namespace std;

template <typename T, typename ... Ts, int N = 1 + sizeof...(Ts)>
array<T, N> do_stuff(const T& arg, const Ts&... args) {
  return array<T, N>{{arg, args...}};
}

int main() {
  auto x = do_stuff(1, 2, 3);
  return 0;
}
