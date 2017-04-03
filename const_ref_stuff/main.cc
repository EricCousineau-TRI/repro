#include "main.h"

#include <vector>
#include <iostream>

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
    : value_(dynamic_cast<C*>(b.value_))
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
    bindings_.push_back(Binding<Base>(value));
    return bindings_.back();
  }
  const Binding<Base>& Add(Child* value) {
    return Add(dynamic_cast<Base*>(value));
  }
private:
  std::vector<Binding<Base>> bindings_;
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
    << "Done" << std::endl;

  return 0;
}
