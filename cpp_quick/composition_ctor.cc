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

class Vars {
public:
    inline Vars(const string& name)
        : vars_{{name}}
    { }
    inline Vars(const Var& var)
        : vars_{var}
    { }

    template<typename T>
    inline Vars(const vector<T>& items)
    {
        for (const auto& item : items)
            append(item);
    }

    // Explicit, unambiguous intiailizer_list's for syntactic sugar
    inline Vars(std::initializer_list<string> list)
        : Vars(vector<string>(list))
    { }
    inline Vars(std::initializer_list<Var> list)
        : Vars(vector<Var>(list))
    { }
    inline Vars(std::initializer_list<Vars> list)
        : Vars(vector<Vars>(list))
    { }

    void append(const Vars& other) {
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

// TODO: Review flexibility...

int main() {
    string var {"a"};
    Vars vars = {"a", "b"}; // {"a"} does not work
    vars.print();
    Vars vars_nest = {vars, vars}; // , var, Var("a")};
    vars_nest.print();
    // Vars vars2("a"); // does not work
    return 0;
}
