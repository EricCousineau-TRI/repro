#include "main.h"

#include <string>
#include <vector>
#include <iostream>
#include <memory>

using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::shared_ptr;
using std::make_shared;

/*
Goal: Consolidate constraint adding

auto AddConstraint(T&& ...)
 b = AddBinding(...);
 ^ Can this be done?

AddLinearConstraint(T&& ...)
 b = AddBinding<LinearConstraint>(...);

*/

auto get_value() {
  return 2;
}

// Definte base, for later specialization
template<typename ... Ts>
auto impl(Ts ... args) {
  return overload_not_implemented(args...);
}

template<typename T>
auto impl(shared_ptr<T> ptr) {
  return string("impl(shared_ptr<T>)");
}

template<>
auto impl(int x, int y) {
  return static_cast<const char*>("impl(x, y)");
}

template<typename ... Ts>
auto variadic_dispatch(Ts ... args) {
  return impl(args...);
}



template<typename C>
class Binding {
public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(Binding)

  Binding(C* value)
    : value_(value)
  { }

  template<typename U>
  Binding(const Binding<U>& b,
    typename std::enable_if<std::is_convertible<
    U*, C*>::value>::type* = nullptr)
    : value_(dynamic_cast<C*>(b.get()))
  { }

  C* get() const {
    return value_;
  }

private:
  C* value_;
};

// Goal: Check to see to what limit we can use "const Binding<T>&" forwarding
class Base { };
class Child : public Base { };

class Container {
public:
  const Binding<Base>& Add(Base* value) {
    base_.push_back(Binding<Base>(value));
    return base_.back();
  }
  const Binding<Child>& Add(Child* value) {
    child_.push_back(Binding<Child>(value));
    return child_.back();
  }
private:
  template<typename C>
  using BindingList = std::vector<Binding<C>>;

  BindingList<Base> base_;
  BindingList<Child> child_;
};

#define PRINT(expr) #expr ": " << (expr) << endl

int main() {
  Base a;
  Child b;

  Container c;
  auto r1 = c.Add(&a);
  auto r2 = c.Add(&b);

  cout
    << PRINT(r1.get())
    << PRINT(r2.get())
    << PRINT(get_value())
    << PRINT(variadic_dispatch(1, 2))
    << PRINT(variadic_dispatch(make_shared<int>(10)))
    << "Done" << endl;

  return 0;
}
