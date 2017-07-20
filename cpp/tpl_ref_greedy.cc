// Goal:
// Figure out how T&& works in forwarding. See if this can be generalized to a greedy is_convertible match
// 
// Purpose:
// In cpp/eigen/matrix_inheritance.cc, Eigen::Matrix's ctor has a the form
//   template<typename T> Matrix(const T& x)
// When there is an attempt to override this, via
//    template<typename T> MyMat(T&& x) { ... }
//    using Matrix::Matrix
// the Matrix version wins.
// It'd be nice to figure out how generalize the child-class ctor to prevent greediness

// @ref http://stackoverflow.com/a/27835982/170413
// Mentions the ability to delegate via `std::forward` to a secondary
// But we do not have that option, as reference collapsing cannot permit delegation to
// a T&& overload since that would cretae a cycle.

#include "name_trait.h"

#include <string>
#include <iostream>

using std::cout;
using std::endl;
using std::string;

struct Base {
    Base() {
        cout << "Base()" << endl;
    }
    template<typename T>
    Base(const T& x) {
        cout << "Base(const T&) [ T = " << name_trait<T>::name() << " ]" << endl;
    }
};

struct Child : public Base {
    template<typename T, typename Cond =
        typename std::enable_if<std::is_convertible<T, int>::value>::type>
    Child(T&& x) {
        cout << "Child(T&&) [ T = " << name_trait<T>::name() << " ]" << endl;
    }
    using Base::Base;
};

struct ChildDirect : public Base {
    template<typename T, typename Cond =
        typename std::enable_if<std::is_convertible<T, int>::value>::type>
    ChildDirect(T&& x) {
        cout << "ChildDirect(T&&) [ T = " << name_trait<T>::name() << " ]" << endl;
    }
    // using Base::Base;
};

struct ChildWithConst : public Base {
    template<typename T, typename Cond =
        typename std::enable_if<std::is_convertible<T, int>::value>::type>
    ChildWithConst(const T& x) {
        cout << "ChildWithConst(const T&) [ T = " << name_trait<T>::name() << " ]" << endl;
    }
    using Base::Base;
};

struct ChildWithOverload : public Base {
    template<typename T, typename Cond =
        typename std::enable_if<std::is_convertible<T, int>::value>::type>
    ChildWithOverload(T&& x) {
        // Cannot delegate to (const T&) case...
        cout << "ChildWithOverload(T&&) [ T = " << name_trait<T>::name() << " ]" << endl;
    }
    template<typename T, typename Cond =
        typename std::enable_if<std::is_convertible<T, int>::value>::type>
    ChildWithOverload(const T& x) {
        cout << "ChildWithOverload(const T&) [ T = " << name_trait<T>::name() << " ]" << endl;
    }
    using Base::Base;
};

int main() {
    int x = 1;
    const int y = 2;
    const double z = 1.5;

    int _ {}; // For indentation

    EVAL({ Child c(1); });
    EVAL({ _; ChildDirect c(1); });
    EVAL({ _; ChildWithConst c(1); });
    EVAL({ _; ChildWithOverload c(1); });
    cout << "---" << endl;

    EVAL({ Child c(x); });
    EVAL({ _; ChildDirect c(x); });
    EVAL({ _; ChildWithConst c(x); });
    EVAL({ _; ChildWithOverload c(x); });
    cout << "---" << endl;

    EVAL({ Child c(y); });
    EVAL({ _; ChildDirect c(y); });
    EVAL({ _; ChildWithConst c(y); });
    EVAL({ _; ChildWithOverload c(y); });
    cout << "---" << endl;

    EVAL({ Child c(z); });
    EVAL({ _; ChildDirect c(z); });
    EVAL({ _; ChildWithConst c(z); });
    EVAL({ _; ChildWithOverload c(z); });
    cout << "---" << endl;
    return 0;
}
