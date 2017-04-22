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
protected:
    string name_;
};

class Vars {
public:
    inline Vars(const Var& var)
        : vars_{var}
    { }
    inline Vars(const vector<Var>& vars)
        : vars_{vars}
    { }
protected:
    vector<Var> vars_;
};

int main() {
    Var var {"a"};
    Vars vars {{"a"}}; // {"a"} does not work
    // Vars vars2("a"); // does not work
    return 0;
}
