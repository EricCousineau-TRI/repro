// Goal: Check that methods can be overloaded and template-specialized unambiguously
// Motivation: https://github.com/RobotLocomotion/drake/issues/5890#issuecomment-298381163

#include <string>
#include <vector>
#include <iostream>
#include <memory>

#include "drake_copy.h"
#include "name_trait.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::shared_ptr;
using std::make_shared;
using std::declval;
using std::dynamic_pointer_cast;

class NameBase {
public:
    virtual string GetName() const = 0;
};

template<typename T>
class NameMixin : public NameBase {
public:
    string GetName() const override { return name_trait<T>::name(); }
};

class Constraint; NAME_TRAIT(Constraint);
class AConstraint; NAME_TRAIT(AConstraint);
class BConstraint; NAME_TRAIT(BConstraint);
template <typename C> class Binding; NAME_TRAIT_TPL(Binding);

/* <snippet from="../variadic_dispatch/main.cc"> */
typedef vector<string> VarList;

template<typename C>
class Binding : public NameMixin<Binding<C>> {
public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(Binding)

  typedef C ContraintType;
  typedef NameMixin<Binding<C>> Base;

  Binding(shared_ptr<C> value, const VarList& var_list)
    : value_(value), var_list_(var_list)
  { }

  template<typename U>
  Binding(const Binding<U>& b, 
    typename std::enable_if<std::is_convertible<U*, C*>::value>::type* = nullptr)
    : value_(dynamic_pointer_cast<C>(b.get()))
  { }

  const shared_ptr<C>& get() const {
    return value_;
  }

  string GetName() const override {
    return "compile-time { " + Base::GetName() + " }, "
        "run-time { Binding<" + value_->GetName() + "> }";
  }

private:
  shared_ptr<C> value_;
  VarList var_list_;
};

class Constraint : public NameMixin<Constraint> { };
class AConstraint : public Constraint {
public:
  AConstraint(int a)
    : a_(a)
  { }
  string GetName() const override { return "AConstraint"; }
private:
  int a_;
};
class BConstraint : public Constraint {
public:
  BConstraint(int b)
    : b_(b)
  { }
  string GetName() const override { return "BConstraint"; }
private:
  int b_;
};

template<typename C>
using BindingList = std::vector<Binding<C>>;
/* </snippet> */


template<typename C, typename ... Args>
auto CreateBinding(const shared_ptr<C>& c, const VarList& vars) {
  return Binding<C>(c, vars);
}

class Expression {
public:
    Expression(int a)
        : a_(a) {}
    int a() const { return a_; }
    VarList GetVars() const { return {"a"}; }
private:
    int a_;
};

// Modified from above snippet
class ConstraintContainer {
public:
  BindingList<Constraint> generic_constraints_;
  BindingList<AConstraint> a_constraints_;
  BindingList<BConstraint> b_constraints_;

  template<typename C>
  auto AddConstraint(shared_ptr<C> ptr, const VarList& vars) {
    auto binding = CreateBinding(ptr, vars);
    return binding;
  }

  Binding<Constraint> AddConstraint(const Expression& e) {
    if (e.a() < 0) {
        auto binding = CreateBinding(make_shared<AConstraint>(e.a()),
                                     e.GetVars());
        return binding;
    } else {
        auto binding = CreateBinding(make_shared<BConstraint>(e.a()),
                                     e.GetVars());
        return binding;
    }
  }

  template <typename C>
  Binding<C> AddConstraint(const Expression& e) {
    // Default
    auto binding = CreateBinding(make_shared<C>(e.a()), e.GetVars());
    return binding;
  }
};

template <>
Binding<BConstraint> ConstraintContainer::AddConstraint<BConstraint>(
    const Expression& e) {
    auto binding = CreateBinding(make_shared<BConstraint>(2 * e.a()),
                                 e.GetVars());
    cout << "Specialized B" << endl;
    return binding;
}

int main() {
    Expression e_a = -5;
    Expression e_b = 5;

    ConstraintContainer c;
    cout
         << PRINT( c.AddConstraint(e_a).GetName() )
         << PRINT( c.AddConstraint(e_b).GetName() )
         << PRINT( c.AddConstraint<AConstraint>(e_a).GetName() )
         << PRINT( c.AddConstraint<BConstraint>(e_b).GetName() );

    return 0;
}
