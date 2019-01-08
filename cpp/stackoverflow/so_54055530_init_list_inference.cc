// Purpose: Try to do min repro: https://stackoverflow.com/a/54056037/170413

#include <cstdint>

#include <utility>
#include <vector>

// From: pybind11:085a294:include/pybind11/buffer_info.h
using ssize_t = size_t;

template <typename T>
class any_container {
 public:
  any_container() = default;

  template <typename It>
  any_container(It first, It last) : v(first, last) { }

  template <
      typename TIn,
      typename = std::enable_if_t<std::is_convertible<TIn, T>::value>>
  any_container(const std::initializer_list<TIn> &c)
      : any_container(c.begin(), c.end()) { }

 private:
  std::vector<T> v;
};

struct buffer_info {
  buffer_info(any_container<ssize_t>) {}
};

// From: armadillo-code:150fd5c
// - include/armadillo_bits/
//   - Mat_bones.hpp
//   - typedef_elem.hpp
typedef unsigned long long u64;
typedef u64 uword;

int main() {
  uword n_rows = 1, n_cols = 2;
  buffer_info({n_rows, n_cols});
  return 0;
}
