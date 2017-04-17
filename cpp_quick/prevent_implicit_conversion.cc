/*
Goal: Provide mechanism to prevent implicit conversion
    (such that const-reference will consistently provide a living pointer)

Refs:
* http://stackoverflow.com/a/12877589/7829525

*/

#include "drake_copy.h"

#include <iostream>
#include <string>
#include <vector>
using std::cout;
using std::endl;
using std::string;
using std::vector;

#include "name_trait.h"

template<typename T>
void overload_info(const T& x) {
  cout << "overload: " << "const " << name_trait<T>::name() << "&" << endl;
}

template<typename T>
void overload_info(T&& x) {
  cout << "overload: " << name_trait<T>::name() << "&&" << endl;
}

// Based on: drake-distro:drake/solvers/binding.h
// Motivation: Will provide implicit copy conversion
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
  {
    cout << "implicit copy: " << name_trait<Binding<U>>::name() << "  ->  " << name_trait<Binding<C>>::name() << endl;
  }

  C* get() const {
    return value_;
  }

private:
  C* value_;
  VarList var_list_;
};
NAME_TRAIT_TPL(Binding);

class Constraint { };
NAME_TRAIT(Constraint);

class LinearConstraint : public Constraint {
public:
  LinearConstraint(int a)
    : a_(a)
  { }
private:
  int a_;
};
NAME_TRAIT(LinearConstraint);

class QuadraticConstraint : public Constraint {
public:
  QuadraticConstraint(int Q, int f)
    : Q_(Q), f_(f)
  { }
private:
  int Q_;
  int f_;
};
NAME_TRAIT(QuadraticConstraint);

#define CALL(x) cout << ">>> " #x << endl; x; cout << endl;

int main() {
  Constraint c;
  LinearConstraint lc(1);
  QuadraticConstraint qc(2, 3);

  Binding<Constraint> bc(&c, {});
  Binding<LinearConstraint> blc(&lc, {});
  Binding<QuadraticConstraint> bqc(&qc, {});

  CALL(overload_info(bc));
  CALL(overload_info<Binding<Constraint>>(bc));
  CALL(overload_info<Binding<LinearConstraint>>(blc));

  CALL(overload_info<Binding<Constraint>>(blc));
  CALL(overload_info<Binding<Constraint>>(bqc));
}
