// Goal: See if it's possible to have a class compose itself, and implicitly accept initializer lists
// Purpose: See if its possible to simplify DecisionVariable stuff
//  Explicitly constrian DecisionVar, such that that is not part of the template generalization

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
        : name_(name)
    { }
    inline string name() const { return name_; }
protected:
    string name_;
};

class VarList {
public:
    // If we provide explicit overloads for the differnet ways we wish to consruct these
    // it does not puke
    inline VarList(const char* name)
        : vars_{{name}}
    { }
    inline VarList(const string& name)
        : vars_{{name}}
    { }
    inline VarList(const Var& var)
        : vars_{var}
    { }

    // Provide generics
    template<typename T>
    inline VarList(const vector<T>& list) {
        for (const auto& item : list)
            append(item);
    }
    // We must expliclitly specify intializer_list compatibility
    template<typename T>
    inline VarList(std::initializer_list<T> list) {
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
    inline VarList(std::initializer_list<VarList> list)
        : VarList(vector<VarList>(list))
    { }

    void append(const VarList& other) {
        for (const auto& v : other.vars_)
            vars_.push_back(v);
    }

    void print(ostream& os = cout) const {
        for (const auto& var : vars_)
            os << var.name() << ", ";
        os << endl;
    }

protected:
    vector<Var> vars_;
};

ostream& operator<<(ostream& os, const VarList &vars) {
    vars.print(os);
    return os;
}


// TODO: Review flexibility...

int main() {
    string var {"a"};
    VarList vars = {"a", "b"};
    cout << vars << endl;
    // VarList vars {"a"};
    vars.print();
    // Heterogeneous (VarList)
    VarList vars_nest = {vars, vars, "a"};
    vars_nest.print();
    


    return 0;
}
