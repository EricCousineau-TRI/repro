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
using std::declval;
// using std::decltype;

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
/* Can't get this to work
template<typename ... Ts>
std::enable_if<false>::type* impl(Ts ... args) {
  return nullptr;
}
*/

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

#define PRINT(expr) #expr ": " << (expr) << endl

void container_stuff();

int main() {
  cout
    << PRINT(get_value())
    << PRINT(variadic_dispatch(1, 2))
    << PRINT(variadic_dispatch(make_shared<int>(10)));
    // << PRINT(variadic_dispatch("bad overload"));

  container_stuff();

  return 0;
}

typedef vector<string> VarList;

template<typename C>
class Binding {
public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(Binding)

  Binding(C* value, const VarList& var_list)
    : value_(value), var_list_(var_list)
  { }

  template<typename U>
  Binding(const Binding<U>& b, 
    typename std::enable_if<std::is_convertible<U*, C*>::value>::type* = nullptr)
    : value_(dynamic_cast<C*>(b.get()))
  { }

  C* get() const {
    return value_;
  }

private:
  C* value_;
  VarList var_list_;
};

// Goal: Check to see to what limit we can use "const Binding<T>&" forwarding
class Constraint { };
class LinearConstraint : public Constraint {
public:
  LinearConstraint(int A, int b)
    : A_(A), b_(b)
  { }
private:
  int A_;
  int b_;
};
class QuadraticConstraint : public Constraint {
  QuadraticConstraint(int Q, int f)
    : Q_(Q), f_(f)
  { }
private:
  int Q_;
  int f_;
};

template<typename C>
using BindingList = std::vector<Binding<C>>;

template<typename ... Ts>
auto create_binding_impl(Ts ... args) {
  return overload_not_implemented(args...);
}

template<>
auto create_binding_impl(int x) {
  return 1;
}

template<typename ... Ts>
class create_binding_traits {
  // typedef std::result_of<create_binding_impl(Ts...)>::type return_type;
  // typedef std::result_of<create_binding_impl, int>::type return_type;
  typedef void* return_type;

  static return_type call(Ts ... args) {
    return invalid_overload(args ...);
  }
};

// template<typename Func, typename ... Args>
// using return_type_of = decltype(Func(declval<Args...>()));
// typedef return_type_of<create_binding_impl, int> return_type_cur;

typedef decltype(create_binding_impl(declval<int>())) return_type_simple;
return_type_simple test_value;

// template<typename F, typename ... Args>
// auto return_type_of(F&& f, Args... args) {
//   typedef decltype(F(decltype<Args...>())) return_type;
//   return declval<return_type>();
// }

// typedef std::return_of<create_binding_impl, int>::type return_type;

// template<>
// class create_binding_traits {
//   typedef Binding<Constraint> return_type;
//   static return_type call(int x, int y) {
//   }
// };

class ConstraintContainer {
public:
  // const Binding<Constraint>& Add(Constraint* value) {
  //   base_.push_back(Binding<Constraint>(value));
  //   return base_.back();
  // }
public:

  BindingList<Constraint> generic_constraints_;
  BindingList<LinearConstraint> linear_constraints_;
  BindingList<QuadraticConstraint> quadratic_constraints_;

  // Can't change return type? Have to use type traits... :(

  template<typename C>
  auto& GetList();
  // template<typename L
  // auto& GetList() { return linear_constraints_; }
  // auto& GetList() { return quadratic_constraints_; }
};

// Can't use auto& :(
template<>
auto& ConstraintContainer::GetList<Constraint>() {
  return generic_constraints_;
}
template<>
auto& ConstraintContainer::GetList<LinearConstraint>() {
  return linear_constraints_;
}
template<>
auto& ConstraintContainer::GetList<QuadraticConstraint>() {
  return quadratic_constraints_;
}

void container_stuff() {
  Constraint a;
  LinearConstraint b(1, 2);

  ConstraintContainer c;

  cout
    << PRINT(&c.GetList<QuadraticConstraint>() == &c.quadratic_constraints_);

  // auto r1 = c.Add(&a);
  // auto r2 = c.Add(&b);

  // cout
  //   << PRINT(r1.get())
  //   << PRINT(r2.get());
}
