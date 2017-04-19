#include <string>
#include <vector>
#include <iostream>
#include <memory>

#include "cpp_quick/drake_copy.h"
#include "cpp_quick/name_trait.h"

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

template<typename ... Args>
using variadic_dispatch_return_type = decltype(variadic_dispatch(declval<Args>()...));
typedef variadic_dispatch_return_type<int, int> basic_return_type;


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

  typedef C ContraintType;

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
  LinearConstraint(int a)
    : a_(a)
  { }
private:
  int a_;
};
class QuadraticConstraint : public Constraint {
public:
  QuadraticConstraint(int Q, int f)
    : Q_(Q), f_(f)
  { }
private:
  int Q_;
  int f_;
};

std::ostream& operator<<(std::ostream& os, const Binding<Constraint>& c) {
  return os << "Constraint";
}
std::ostream& operator<<(std::ostream& os, const Binding<LinearConstraint>& c) {
  return os << "LinearConstraint";
}
std::ostream& operator<<(std::ostream& os, const Binding<QuadraticConstraint>& c) {
  return os << "QuadraticConstraint";
}

template<typename C>
using BindingList = std::vector<Binding<C>>;


template<typename ... Ts>
auto create_binding_impl(Ts ... args) {
  return overload_not_implemented(args...);
}
template<>
auto create_binding_impl(int x, const VarList& var_list) {
  // Will be LinearConstriant
  return Binding<LinearConstraint>(new LinearConstraint(x), var_list);
}
template<>
auto create_binding_impl(int x, int y) {
  // By default will be quadratic constraint
  return Binding<QuadraticConstraint>(new QuadraticConstraint(x, y), {"x"});
}

template<typename C, typename ... Args>
auto create_binding_specific_impl(Args ... args) {
  return Binding<C>(args...);
}


// // This would be redundant - would need to specialize template twice :(
// template<typename ... Ts>
// class create_binding_traits {
//   // typedef std::result_of<create_binding_impl(Ts...)>::type return_type;
//   // typedef std::result_of<create_binding_impl, int>::type return_type;
//   typedef void* return_type;

//   static return_type call(Ts ... args) {
//     return invalid_overload(args ...);
//   }
// };

// template<typename Func, typename ... Args>
// using return_type_of = decltype(Func(declval<Args...>()));
// typedef return_type_of<create_binding_impl, int> return_type_cur;

// typedef decltype(create_binding_impl(declval<int>())) return_type_simple;
// // return_type_simple test_value;

// // This isn't really needed
// template<typename ... Args>
// using create_binding_return_type = decltype(create_binding_impl(declval<Args>()...));

// create_binding_return_type<int, int> other_value;

// // Note sure how to make this work
// template<typename F, typename ... Args>
// using return_type_complex = decltype(F(declval<Args>()...));
// return_type_complex<decltype(create_binding_impl), int, int> other_value;

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

  template<typename ... Args>
  auto AddConstraint(Args ... args, const VarList& vars) {
    auto binding = create_binding_impl(args..., vars);
    // auto list = GetList<decltype<binding>();
    return binding;
  }
  template<typename C, typename ... Args>
  auto AddConstraintSpecific(Args ... args) {
    auto binding = create_binding_specific_impl<C>(args...);
  }

  template<typename ... Args>
  auto AddLinearConstraint(Args ... args) {
    return AddConstraintSpecific<LinearConstraint, Args...>(args...);
  }
};

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



string quick_check(const vector<string>& test_list) {
  return "yup";
}
template<typename T>
auto tpl_check(T value) {
  return 0;
}
template<>
auto tpl_check(const vector<string>& test_list) {
  return "yup";
}
template<>
auto tpl_check(std::initializer_list<string> list) {
  return "yup init";
}
template<>
auto tpl_check(int x) {
  return "nope";
}


void container_stuff() {
  Constraint a;
  LinearConstraint b(1);
  QuadraticConstraint b2(1, 2);

  ConstraintContainer c;
  
  // Can deduce as such
  // cout << quick_check({"hello"}) << endl << tpl_check(vector<string>{"good"}) << endl;
    //tpl_check({"good"}) << endl; // Can't infer T for initializer list

  c.AddConstraint(1, VarList {string("x")});

  cout
    // << PRINT(&c.GetList<QuadraticConstraint>() == &c.quadratic_constraints_)
    // << PRINT(c.AddConstraint(1, {string("x")})) // Can't do just {"x"}
    // << PRINT(c.AddConstraint(1, 2));
    ;

  // auto r1 = c.Add(&a);
  // auto r2 = c.Add(&b);

  // cout
  //   << PRINT(r1.get())
  //   << PRINT(r2.get());
}
