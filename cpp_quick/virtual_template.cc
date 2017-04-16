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

template<typename T>
struct trait {
    static inline const char* name() { return "generic"; }
};
template<>
struct trait<int> {
    static inline const char* name() { return "int"; }
};
template<>
struct trait<double> {
    static inline const char* name() { return "double"; }
};

class Base {
public:
    virtual void tpl(int x) = 0;
    virtual void tpl(double x) = 0;
};

template<typename Base, typename Derived>
class BaseImpl : public Base {
public:
    void tpl(int x) override {
        derived()->tpl_impl(x);
    }
    void tpl(double x) override {
        derived()->tpl_impl(x);
    }
private:
    inline Derived* derived() {
        return static_cast<Derived*>(this);
    }
    inline const Derived* derived() const {
        return static_cast<const Derived*>(this);
    }
};

class Child : public BaseImpl<Base, Child> {
protected:
    friend class BaseImpl<Base, Child>;
    template<typename T>
    void tpl_impl(T x) {
        cout << "Child::tpl_impl<" << trait<T>::name() << ">(" << x << ")" << endl;
    }
};

int main() {
    Child c;
    Base *b = &c;
    b->tpl(2);
    b->tpl(3.0);
    return 0;
}
