#include <type_traits>

class A {
 public:
  static constexpr int value = 10;
};

class B {};

// @ref https://stackoverflow.com/questions/10178598/specializing-a-templated-member-of-a-template-class
template <typename T>
class Example {
 public:
  int Stuff() {
    return stuff_helper<kIsA>::impl();
  }
 private:
  static constexpr bool kIsA = std::is_base_of<A, T>::value;

  // Use `Extra` to permit this to be partially specialized, such that
  // we can define it within the class.
  template <bool kIsA_, typename Extra = void>
  struct stuff_helper {
    static int impl() {
      return -1;
    }
  };
  template <typename Extra>
  struct stuff_helper<true, Extra> {
    static int impl() {
      return T::value;
    }
  };
};

int extra_stuff();
