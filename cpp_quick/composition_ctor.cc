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
    // inline VarList(std::initializer_list<VarList> list)
    //     : VarList(vector<VarList>(list))
    // { }

    void append(const VarList& other) {
        for (const auto& v : other.vars_)
            vars_.push_back(v);
    }

    void print() {
        for (const auto& var : vars_)
            cout << var.name() << ", ";
        cout << endl;
    }

protected:
    vector<Var> vars_;
};

// template
// VarList::VarList(std::initializer_list<string> list);

// TODO: Review flexibility...

int main() {
    string var {"a"};
    VarList vars = {"a", "b"}; // {"a"} does not work
    vars.print();
    VarList vars_nest = {vars, vars}; // , var, Var("a")};
    vars_nest.print();
    // VarList vars2("a"); // does not work
    return 0;
}
