#include <iostream>
#include <string>

#include "name_trait.h"

using namespace std;

template<typename Derived>
struct Base {
protected:
    Base() = default;
public:
    Derived& derived() { return static_cast<Derived&>(*this); }

    void dofunc() {
        derived().future_function();
    }

    Derived operator+(const Derived& b) {
        Derived c(derived());
        cout << " + " << endl;
        return c += b;
    }
};

struct Child : Base<Child>
{
    // using Base::Base;
    Child() = default;

    // BUG? Will complain about delete rvalue ctor, but if we implement it explicitly, it does not get called...
    Child(Child&&) {
        cout << "using rvalue" << endl;
    }
    // Child(Child&&) = delete;
    
    Child(const Child&) {
        cout << "using const lvalue" << endl;
    }

    void call() {
        dofunc();
    }

    void future_function() {
        cout << "future" << endl;
    }

    Child& operator+=(const Child& other) {
        cout << " += " << endl;
        return *this;
    }
};



// using std::cout;
// using std::endl;
// using std::string;

void my_func(string& s) {
    cout << "lvalue reference" << endl;
}

void my_func(string&& s) {
    cout << "rvalue reference" << endl;
}

void my_func(const string& s) {
    cout << "const lvalue reference" << endl;
}

void my_func(const string&& s) {
    cout << "const rvalue reference" << endl;
}

// auto& other_func(bool value) {
//     if (value)
//         return "lkjsf";
//     else
//         return "lajasldkjf";
// }

int main() {
    string x = "string";
    const string cx = "const string";
    EVAL(my_func(x));
    EVAL(my_func(static_cast<const string>(string("wut"))));
    EVAL(my_func("hello"));
    EVAL(my_func(cx));
    auto s = "hello";
    auto& sr = "hello";
    cout
         << PRINT(name_trait<decltype(s)>::name())
         << PRINT(name_trait<decltype(sr)>::name())
         << PRINT(name_trait<const int&&>::name());

    Child c1;
    Child c2;
    Child c3 = c1 + c2;
    c3.call();

    return 0;
}
