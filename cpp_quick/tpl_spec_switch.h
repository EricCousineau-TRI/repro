#include <type_traits>

class A {
 public:
  static constexpr int value = 10;
};

class B {};

namespace detail {

template <typename U, bool kIsA>
struct stuff_helper {
  static int DoStuff(U* obj) {
    return -1;
  }
};
template <typename U>
struct stuff_helper<U, true /* kIsA */> {
  static int DoStuff(U* obj) {
    return obj->private_value();
  }
};

}  // namespace detail

template <typename T>
class Base {
 protected:
  int private_value() {
    return T::value;
  }
};

// @ref https://stackoverflow.com/questions/10178598/specializing-a-templated-member-of-a-template-class
template <typename T>
class Example : public Base<T> {
 public:
  int Stuff() {
    return stuff_helper::DoStuff(this);
  }
 private:
  static constexpr bool kIsA = std::is_base_of<A, T>::value;
  using stuff_helper = detail::stuff_helper<Example, kIsA>;
  // Permit the helper to have access
  friend stuff_helper;
};

int extra_stuff();
