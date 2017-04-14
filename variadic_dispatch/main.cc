#include "main.h"

#include <vector>
#include <iostream>

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

int main() {
  Base a;
  Child b;

  Container c;
  auto r1 = c.Add(&a);
  auto r2 = c.Add(&b);

  std::cout
    << "r1: " << r1.get() << std::endl
    << "r2: " << r2.get() << std::endl
    << "value: " << get_value() << std::endl
    << "Done" << std::endl;

  return 0;
}
