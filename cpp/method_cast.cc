#include <iostream>

using std::cout;

class Base {
 public:
  Base(int value) : value_(value) {}
  virtual ~Base() = default;

 protected:
  int ProtectedMethod(int x) const {
    return x + value_;
  }

 private:
  int value_{};
};

class Child : public Base {
 public:
  using Base::Base;
  ~Child() = default;
};

class BasePublic : public Base {
 public:
  int OtherMethod(int x) const {
    return x + hot_garbage_;
  }
  using Base::ProtectedMethod;
 
 private:
  int hot_garbage_{1000};
};

using BaseMethod = int (Base::*)(int) const;

template <typename Return, typename... Args>
struct overload_cast_impl {
  auto operator()(Return (*func)(Args...)) const { return func; }

  template <typename Class>
  auto operator()(Return (Class::*method)(Args...)) const {
    return method;
  }

  template <typename Class>
  auto operator()(Return (Class::*method)(Args...) const) const {
    return method;
  }
};

template <typename Return, typename... Args>
constexpr auto overload_cast_explicit = overload_cast_impl<Return, Args...>{};

int main() {
  Child child(10);

  BaseMethod protected_method = &BasePublic::ProtectedMethod;
  std::cout << (child.*protected_method)(1) << "\n";

  BaseMethod other_method =
      // &BasePublic::OtherMethod;  // Does not compile.
      // overload_cast_explicit<int, int>(&BasePublic::OtherMethod);  // Does not compile
      static_cast<BaseMethod>(&BasePublic::OtherMethod);  // Compiles :(
  std::cout << (child.*other_method)(1) << "\n";

  return 0;
}
