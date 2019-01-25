#include <cassert>

#include <functional>
#include <iostream>
#include <memory>
#include <utility>

template <typename Base, typename Return, typename... Args>
class FunctionOrMethod {
 public:
  using Function = std::function<Return (Args...)>;

  template <typename Derived>
  using ConstMethod = Return (Derived::*)(Args...) const;

  template <typename Lambda>
  FunctionOrMethod(Lambda lambda)
    : function_(lambda) {}

  template <typename Derived>
  FunctionOrMethod(ConstMethod<Derived> const_method)
    : method_(new MethodImpl<const Derived, ConstMethod<Derived>>(
          const_method)) {}

  Function get(Base* ptr) const {
    if (function_) {
      return function_;
    } else {
      return method_->BindFront(ptr);
    }
  }

 private:
  class MethodErasure {
   public:
    virtual ~MethodErasure() {}
    virtual Function BindFront(Base* ptr) = 0;
  };

  template <typename Derived, typename Method>
  class MethodImpl : public MethodErasure {
   public:
    static_assert(std::is_base_of<Base, Derived>::value, "Invalid inheritance");
    MethodImpl(Method method) : method_(method) {}

    Function BindFront(Base* base) override {
      Derived* ptr = dynamic_cast<Derived*>(base);
      assert(ptr != nullptr);
      auto is_void = std::is_same<Return, void>{};
      return BindFrontImpl(is_void, ptr, method_);
    }

   private:
    auto BindFrontImpl(
        std::true_type /* is_void */, Derived* ptr, Method method) {
      return [ptr, method](Args... args) {
        (ptr->*method)(std::forward<Args>(args)...);
      };
    }

    auto BindFrontImpl(
        std::false_type /* is_void */, Derived* ptr, Method method) {
      return [ptr, method](Args... args) -> Return {
        return (ptr->*method)(std::forward<Args>(args)...);
      };
    }

    Method method_;
  };

  Function function_;
  std::unique_ptr<MethodErasure> method_;
};

struct Base {
  virtual ~Base() {}

  template <typename Return, typename... Args>
  using Callback = FunctionOrMethod<Base, Return, Args...>;

  auto DeclareStuff(Callback<void, int> func) {
    return func.get(this);
  }

  auto DeclareMore(Callback<int> func) {
    return func.get(this);
  }
};

struct Example : public Base {
  Example(int value_in) : value(value_in) {}

  void ExampleStuff(int x) const {
    std::cout << "Value Mult: " << x * value << std::endl;
  }

  int ExampleMore() const {
    return value;
  }

  int value{};
};

#define EVAL(x) std::cout << ">>> " #x ";" << std::endl; x; std::cout << std::endl

int main() {
  Example ex(10);

  auto stuff_method = ex.DeclareStuff(&Example::ExampleStuff);
  EVAL(stuff_method(3));
  auto stuff_func = ex.DeclareStuff(
      [](int x) { std::cout << "Func: " << x << std::endl; });
  EVAL(stuff_func(3));

  auto more_method = ex.DeclareMore(&Example::ExampleMore);
  EVAL(std::cout << more_method() << std::endl);
  auto more_func = ex.DeclareMore([]() { return 100; });
  EVAL(std::cout << more_func() << std::endl);

  // Example of bad inheritance.
  struct BadInheritance : public Base {
    void BadStuff(int) const {}
  };
  // ex.DeclareStuff(&BadInheritance::BadStuff);  // Assertion fails.
  return 0;
}

/*
Output:

>>> stuff_method(3);
Value Mult: 30

>>> stuff_func(3);
Func: 3

>>> std::cout << more_method() << std::endl;
10

>>> std::cout << more_func() << std::endl;
100
*/
