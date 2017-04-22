// Goal: See if it's possible to have a class compose itself, and implicitly accept initializer lists
// Purpose: See if its possible to simplify DecisionVariable stuff
//  Explicitly constrain DecisionVar, such that that is not part of the template generalization

#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "name_trait.h"

using std::string;
using std::cout;
using std::endl;
using std::vector;
using std::ostream;

class Var {
public:
    inline Var(const string& name)
        : name_(name) {
        cout << "  Var(string): " << name << endl;
    }
    inline string name() const { return name_; }
protected:
    string name_;
};

class VarList {
public:
    // If we provide explicit overloads for the differnet ways we wish to consruct these
    // it does not puke
    inline VarList(const char* name)
        : vars_{{name}} {
        cout << "VarList(const char*): " << name << endl;
    }
    inline VarList(const string& name)
        : vars_{{name}} {
        cout << "VarList(string): " << name << endl;
    }
    inline VarList(const Var& var)
        : vars_{var} {
        cout << "VarList(Var): " << var.name() << endl;
    }

    // Provide generics
    template<typename T>
    inline VarList(const vector<T>& list) {
        cout << "VarList(vector<T>)" << endl;
        for (const auto& item : list)
            append(item);
    }
    // We must expliclitly specify intializer_list compatibility
    template<typename T>
    inline VarList(std::initializer_list<T> list) {
        cout << "VarList(initializer_list<T>)" << endl;
        for (const auto& item : list)
            append(item);
    }

    // // Explicit, unambiguous intiailizer_list's for syntactic sugar
    // inline VarList(std::initializer_list<string> list)
    //     : VarList(vector<string>(list))
    // { }
    // inline VarList(std::initializer_list<Var> list)
    //     : VarList(vector<Var>(list))
    // { }
    inline VarList(std::initializer_list<VarList> list) {
        cout << "VarList(initializer_list<VarList>): " << endl;
        for (const auto& item : list)
            append(item);
    }

    void append(const VarList& other) {
        for (const auto& v : other.vars_)
            vars_.push_back(v);
    }

    void print(ostream& os = cout) const {
        bool is_first = true;
        os << "(";
        for (const auto& var : vars_)
        {
            if (is_first)
                is_first = false;
            else
                os << ", ";
            os << var.name();
        }
        os << ")";
    }

protected:
    vector<Var> vars_;
};

ostream& operator<<(ostream& os, const VarList &vars) {
    vars.print(os);
    return os;
}

void print(const VarList& vars) {
    cout << vars;
}

int main() {
    // Strings
    EVAL(( print("a") ));
    EVAL(( print({"a"}) ));
    EVAL(( print({"a", "b"}) ));
    EVAL(( print({string("c")}) ));    

    cout << "[ Heterogeneous ]" << endl;
    const char* cstr = "cstr";
    string str("str");
    Var var("var");
    VarList vars = {"var[0]", "var[1]", "var[2]"};
    cout << endl << "[ - VarList + other ]" << endl;
    EVAL(( print(vars) ));
    EVAL(( print({vars, vars}) ));
    EVAL(( print({vars, var}) ));
    EVAL(( print({vars, str}) ));
    EVAL(( print({vars, cstr}) ));

    cout << endl << "[ - Var + other ]" << endl;
    EVAL(( print(var) ));
    EVAL(( print({var, vars}) ));
    EVAL(( print({var, var}) ));
    EVAL(( print({var, str}) ));
    EVAL(( print({var, cstr}) ));

    cout << endl << "[ - string + other ]" << endl;
    EVAL(( print(str) ));
    EVAL(( print({str, vars}) ));
    EVAL(( print({str, var}) ));
    EVAL(( print({str, str}) ));
    EVAL(( print({str, cstr}) ));

    cout << endl << "[ - const char* + other ]" << endl;
    EVAL(( print(cstr) ));
    EVAL(( print({cstr, vars}) ));
    EVAL(( print({cstr, var}) ));
    EVAL(( print({cstr, str}) ));
    EVAL(( print({cstr, cstr}) ));

    cout << endl << "[ Composition ]" << endl;
    EVAL(( print({var, {vars, vars, {"a", vars}}}) ));
    return 0;
}
