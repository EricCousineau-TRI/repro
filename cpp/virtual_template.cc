#include <iostream>
using std::cout;
using std::endl;

// Most likely can't have virtual template functions, but I want to see the error
// Yup, can't:
/*
cpp_quick/virtual_template.cc:16:5: error: 'virtual' cannot be specified on member function templates
    virtual void tpl();
    ^~~~~~~~
*/

// class Base {
// public:
//     template<typename T>
//     virtual void tpl(T x);
// };

// Adaptation of Visitor Pattern / CRTP from:
// http://stackoverflow.com/a/5872633/170413

class Base {
public:
    virtual void tpl(int x) = 0;
    virtual void tpl(double x) = 0;
};

// Generics for display
template<typename T>
struct trait {
    static inline const char* name() { return "T"; }
};
template<>
struct trait<int> {
    static inline const char* name() { return "int"; }
};
template<>
struct trait<double> {
    static inline const char* name() { return "double"; }
};

// Use CRTP for dispatch
// Also specify base type to allow for multiple generations
template<typename BaseType, typename DerivedType>
class BaseImpl : public BaseType {
public:
    void tpl(int x) override {
        derived()->tpl_impl(x);
    }
    void tpl(double x) override {
        derived()->tpl_impl(x);
    }
private:
    // Eigen-style
    inline DerivedType* derived() {
        return static_cast<DerivedType*>(this);
    }
    inline const DerivedType* derived() const {
        return static_cast<const DerivedType*>(this);
    }
};

// Have Child extend indirectly from Base
class Child : public BaseImpl<Base, Child> {
protected:
    friend class BaseImpl<Base, Child>;
    template<typename T>
    void tpl_impl(T x) {
        cout << "Child::tpl_impl<" << trait<T>::name() << ">(" << x << ")" << endl;
    }
};

// Have SubChild extend indirectly from Child
class SubChild : public BaseImpl<Child, SubChild> {
protected:
    friend class BaseImpl<Child, SubChild>;
    template<typename T>
    void tpl_impl(T x) {
        cout << "SubChild::tpl_impl<" << trait<T>::name() << ">(" << x << ")" << endl;
    }
};


template<typename BaseType>
void example(BaseType *p) {
    p->tpl(2);
    p->tpl(3.0);
}

int main() {
    Child c;
    SubChild sc;

    // Polymorphism works for Base as base type
    example<Base>(&c);
    example<Base>(&sc);
    // Polymorphism works for Child as base type
    example<Child>(&sc);
    return 0;
}
