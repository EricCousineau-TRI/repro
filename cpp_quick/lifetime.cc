// Goal: Empirically determine lifetimes
// Purpose: Too lazy to fully parse:
// @ref http://en.cppreference.com/w/cpp/language/lifetime

#include <iostream>
#include <utility>

using std::cout;
using std::endl;

#define EVAL(x) \
    std::cout << ">>> " #x ";" << std::endl; \
    x; \
    cout << std::endl
#define EVAL_SCOPED(x) \
    std::cout << ">>> scope { " #x " ; }" << std::endl; \
    { \
        x; \
        std::cout << "   <<< [ exiting scope ]" << std::endl; \
    } \
    std::cout << std::endl

template <int T>
class Lifetime {
public:
    Lifetime() {
        cout << T << ": ctor ()" << endl;
    }
    Lifetime(const Lifetime&) {
        cout << T << ": copy ctor (const lvalue)" << endl;
    }
    Lifetime(Lifetime&& other) {
        cout << T << ": copy ctor (rvalue)" << endl;
        other.moved = true;
    }
    template <int U>
    Lifetime(const Lifetime<U>&) {
        cout
             << T << ": ctor (const Lifetime<" << U << ">&)" << endl;
    }
    template <int U>
    Lifetime(Lifetime<U>&& other) {
        cout
             << T << ": ctor (Lifetime<" << U << ">&&)" << endl;
        other.moved = true;
    }
    ~Lifetime() {
        cout << T << ": dtor";
        if (moved)
            cout << " <-- was moved";
        cout << endl;
    }

    bool moved {false};
protected:
    using Base = Lifetime<T>;
};

void func_in_const_lvalue(const Lifetime<3>&) {
    cout << "func_in_const_lvalue" << endl;
}

Lifetime<3> func_out_value() {
    cout << "func_out_value" << endl;
    return Lifetime<3>();
}

const Lifetime<3>& func_out_const_lvalue() {
    return Lifetime<3>();
}

const Lifetime<3>& func_thru_const_lvalue(const Lifetime<3>& in) {
    cout << "func_thru_const_lvalue" << endl;
    return in;
}
const Lifetime<3>& func_thru_rvalue(const Lifetime<3>& in) {
    cout << "func_thru_rvalue" << endl;
    return in;
}

void section(const char* name) {
    cout << endl << "--- " << name << " ---" << endl << endl;
}

int main() {
    section("Standard");
    EVAL(Lifetime<1> obj1{}; Lifetime<2> obj2{} );
    EVAL_SCOPED( Lifetime<1>(); Lifetime<2>() );
    EVAL_SCOPED( Lifetime<2> copy = obj2 );
    EVAL_SCOPED( Lifetime<2> copy = obj1 );
    EVAL_SCOPED( Lifetime<2> copy = Lifetime<1>() );

    section("In: const T&");
    EVAL( func_in_const_lvalue(Lifetime<3>()) );
    
    section("Out: T");
    EVAL( func_out_value() );
    EVAL_SCOPED( const Lifetime<3>& ref = func_out_value() );
    EVAL_SCOPED( Lifetime<3>&& ref = func_out_value() );

    section("Out: const T&");
    EVAL( func_out_const_lvalue() );
    EVAL_SCOPED( const Lifetime<3>& ref = func_out_const_lvalue() );

    section("Thru: const T&");
    EVAL( func_thru_const_lvalue(Lifetime<3>()) );
    EVAL_SCOPED( const auto& ref = func_thru_const_lvalue(Lifetime<3>()) );

    section("Thru: T&&");
    EVAL( func_thru_rvalue(Lifetime<3>()) );
    EVAL_SCOPED( const auto& ref = func_thru_rvalue(Lifetime<3>()) );

    return 0;
}
