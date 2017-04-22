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
        : names_{{s}}
    { }
    template<typename T>
    inline Vars(const vector<T>& items)
    {
        for (const auto& item : items)
            append(item);
    }
    // Explicit intiailizer_list for syntactic sugar
    inline Vars(std::initializer_list<string> list)
        : Vars(vector<string>(list))
    { }
    inline Vars(std::initializer_list<Vars> list)
        : Vars(vector<Vars>(list))
    { }

    void print() {
        for (const auto& name : names_)
            cout << name << ", ";
        cout << endl;
    }

protected:
    vector<string> names_;

    void append(const Vars& vs) {
        for (const auto& v : vs.names_)
            names_.push_back(v);
    }
};

// TODO: Review flexibility...

int main() {
    string var {"a"};
    Vars vars = {"a", "b"}; // {"a"} does not work
    vars.print();
    Vars names_nest = {vars, vars}; // , var, Var("a")};
    names_nest.print();
    // Vars vars2("a"); // does not work
    return 0;
}
