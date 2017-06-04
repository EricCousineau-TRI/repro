#include <type_traits>

// @ref https://stackoverflow.com/questions/10178598/specializing-a-templated-member-of-a-template-class

class A {
 public:
  static constexpr int value = 10;
};

class B {};


template <typename T>
class Base {
 protected:
  int private_value() {
    return T::value;
  }
};

template <typename T>
class DetailedBase : public Base<T> {
 public:
  int Stuff() {
    return helper::DoStuff(this);
  }
 protected:

  // Place helper specializations inside class to permit friendly
  // access to Base<T>::private_value().
  template <bool kIsA, typename = void>
  struct helper_impl {};

  template <typename Extra>
  struct helper_impl<false, Extra> {
    // Note that we use DetailBase to access Base<T>::private_value.
    // If we passed Base<T>, we would get scope access errors.
    static int DoStuff(DetailedBase* obj) {
      return -1;
    }
  };

  template <typename Extra>
  struct helper_impl<true, Extra> {
    static int DoStuff(DetailedBase* obj) {
      return obj->private_value();
    }
  };

  static constexpr bool kIsA = std::is_base_of<A, T>::value;
  using helper = helper_impl<kIsA>;
};

template <typename T>
class Example : public DetailedBase<T> {
 public:
  int Stuff() {
    return 2 * Base::Stuff();
  }
 protected:
  using Base = DetailedBase<T>;
};

int extra_stuff();
