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
        cout << "ctor (default): " << T << endl;
    }
    Lifetime(const Lifetime&) {
        cout << "copy ctor (const lvalue): " << T << endl;
    }
    Lifetime(Lifetime&&) {
        cout << "copy ctor (rvalue): " << T << endl;
    }
    template <int U>
    Lifetime(const Lifetime<U>&) {
        cout
             << "ctor (const Lifetime<" << U << ">&): "
             << T << endl;
    }
    template <int U>
    Lifetime(Lifetime<U>&&) {
        cout
             << "ctor (Lifetime<" << U << ">&&): "
             << T << endl;
    }
    ~Lifetime() {
        cout << "dtor: " << T << endl;
    }
protected:
    using Base = Lifetime<T>;
};

void func_in_const_lvalue(const Lifetime<1>&) {
    cout << "func_in_const_lvalue" << endl;
}

Lifetime<1> func_out_value() {
    cout << "func_out_value" << endl;
    return Lifetime<1>();
}

const Lifetime<1>& func_thru_const_lvalue(const Lifetime<1>& in) {
    cout << "func_thru_const_lvalue" << endl;
    return in;
}
const Lifetime<1>& func_thru_rvalue(const Lifetime<1>& in) {
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
    EVAL_SCOPED( func_in_const_lvalue(Lifetime<2>()) );
    
    section("Out: T");
    EVAL_SCOPED( func_out_value() );
    EVAL_SCOPED( const Lifetime<1>& ref = func_out_value() );
    EVAL_SCOPED( Lifetime<1>&& ref = func_out_value() );

    section("Thru: const T&");
    EVAL_SCOPED( func_thru_const_lvalue(Lifetime<2>()) );
    EVAL_SCOPED( const auto& ref = func_thru_const_lvalue(Lifetime<2>()) );

    section("Thru: T&&");
    EVAL_SCOPED( func_thru_rvalue(Lifetime<2>()) );
    EVAL_SCOPED( const auto& ref = func_thru_rvalue(Lifetime<2>()) );

    return 0;
}
