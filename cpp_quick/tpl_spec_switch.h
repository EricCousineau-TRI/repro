#include <type_traits>

class A {
 public:
  static constexpr int value = 10;
};

class B {};

// Use `Extra` to permit this to be partially specialized, such that
// we can define it within the class.
template <typename U, bool kIsA_>
struct stuff_helper {
  static int impl(U* obj) {
    return -1;
  }
};
template <typename U>
struct stuff_helper<U, true> {
  static int impl(U* obj) {
    return obj->private_value();
  }
};

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
    return helper::impl(this);
  }
 private:
  static constexpr bool kIsA = std::is_base_of<A, T>::value;
  using helper = stuff_helper<Example, kIsA>;
  // Permit the helper to have access
  friend helper;
};

int extra_stuff();
