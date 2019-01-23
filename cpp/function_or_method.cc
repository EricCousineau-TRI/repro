#include <functional>
#include <iostream>
#include <utility>

template <typename Class, typename Return, typename... Args>
class FunctionOrMethod {
 public:
  using Function = std::function<Return (Args...)>;
  using ConstMethod = Return (Class::*)(Args...) const;

  template <typename Lambda>
  FunctionOrMethod(Lambda lambda)
    : function_(lambda) {}

  FunctionOrMethod(ConstMethod const_method)
    : const_method_(const_method) {}

  Function get(Class* ptr) const {
    if (function_) {
      return function_;
    } else {
      auto is_void = std::is_same<Return, void>{};
      return BindFront(is_void, ptr);
    }
  }

 private:
  auto BindFront(std::true_type /* is_void */, Class* ptr) const {
    ConstMethod const_method = const_method_;
    return [ptr, const_method](Args... args) {
      (ptr->*const_method)(std::forward<Args>(args)...);
    };
  }

  auto BindFront(std::false_type /* is_void */, Class* ptr) const {
    ConstMethod const_method = const_method_;
    return [ptr, const_method](Args... args) {
      return (ptr->*const_method)(std::forward<Args>(args)...);
    };
  }

  Function function_;
  ConstMethod const_method_{};
};

struct Example {
  Example(int value_in) : value(value_in) {}

  template <typename... Extra>
  using Callback = FunctionOrMethod<Example, Extra...>;

  auto DeclareStuff(Callback<void, int> func) {
    return func.get(this);
  }

  void ExampleStuff(int x) const {
    std::cout << "Value: " << x * value << std::endl;
  }

  auto DeclareMore(Callback<int> func) {
    return func.get(this);
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
  return 0;
}

/*
Output:

>>> stuff_method(3);
Value: 30

>>> stuff_func(3);
Func: 3

>>> std::cout << more_method() << std::endl;
10

>>> std::cout << more_func() << std::endl;
100
*/
