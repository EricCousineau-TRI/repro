#include <type_traits>

class A {
 public:
  static constexpr int value = 10;
};

template <typename T>
class Example {
 public:
  int Stuff() {
    return DoStuff<>();
  }
 private:
  static constexpr bool kIsA = std::is_base_of<A, T>::value;

  template <bool kIsA_ = kIsA>
  int DoStuff();
};

template <typename T, bool kIsA_ = true>
inline int Example<T>::DoStuff<kIsA_>() {
  return T::value;
}

template <typename T>
inline int Example<T>::DoStuff<false>() {
  return 1;
}

int extra_stuff();
