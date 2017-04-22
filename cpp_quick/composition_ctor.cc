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
    inline Vars(const string& s)
        : vars_{{s}}
    { }
    inline Vars(const Var& var)
        : vars_{var}
    { }
    template<typename T>
    inline Vars(const vector<T>& items)
    {
        for (const auto& item : items)
            add_var(item);
    }
    // Explicit intiailize_list
    inline Vars(std::initializer_list<string> list)
        : Vars(vector<string>(list))
    { }

    void print() {
        for (const auto& v : vars_)
            cout << v.name() << ", ";
        cout << endl;
    }

protected:
    vector<Var> vars_;
    // void add_var(const Var& v) {
    //     vars_.push_back(v);
    // }
    void add_var(const Vars& vs) {
        for (const auto& v : vs.vars_)
            vars_.push_back(v);
    }
};

// TODO: Review flexibility...

int main() {
    Var var {"a"};
    Vars vars = {"a", "b"}; // {"a"} does not work
    vars.print();
    Vars vars_nest = {vars, vars}; // , var, Var("a")};
    vars_nest.print();
    // Vars vars2("a"); // does not work
    return 0;
}
