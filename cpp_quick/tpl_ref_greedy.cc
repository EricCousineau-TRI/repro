// Goal: Figure out how T&& works in forwarding. See if this can be generalized to a greedy is_convertible match
// Purpose: In eigen_scratch/matrix_inheritance.cc, Eigen::Matrix's ctor has a the form
//   template<typename T> Matrix(const T& x)
// When there is an attempt to override this, via
//    template<typename T> MyMat(T&& x) { ... }
//    using Matrix::Matrix
// the Matrix version wins.
// It'd be nice to figure out how generalize the child-class ctor to prevent greediness

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
    template<typename T>
    Child(T&& x) {
        cout << "Child(T&&) [ T = " << name_trait<T>::name() << " ]" << endl;
    }
    using Base::Base;
};

int main() {
    int x = 1;
    const int y = 2;
    EVAL({ Child c(1); });
    // EVAL( Child(x) );
    // EVAL( CHild(y) );
    return 0;
}
