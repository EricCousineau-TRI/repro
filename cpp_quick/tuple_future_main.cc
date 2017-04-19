#include <iostream>
#include <vector>
#include <string>
#include <sstream>

#include "tuple_future.h"

using std::cout;
using std::endl;
using std::vector;
using std::pair;
using std::string;
using std::ostringstream;

#define PRINT(x) ">>> " #x << std::endl << (x) << std::endl
#define VARIADIC_CALLABLE(func, capture) \
    [capture] (auto&&... args) { \
        return func(std::forward<decltype(args)>(args)...); \
    }


double my_func(int x, double y) {
    cout << "func(int, double)" << endl;
    return x + y;
}

double my_func(double x, int y) {
    cout << "[ reversed ] func(double, int)" << endl;
    return x - y;
}

// Simple: Call with a tuple
auto my_func_callable = [] (auto&& ... args) {
    return my_func(std::forward<decltype(args)>(args)...);
};

void simple_example() {
    auto t = std::make_tuple(1, 2.0);
    cout
        << PRINT(stdfuture::apply(my_func_callable, t))
        << PRINT(stdcustom::apply_reversed(my_func_callable, t));
}

// Advanced: Use make_callable_reversed
void advanced_example() {
    auto my_func_reversed =
        stdcustom::make_callable_reversed(my_func_callable);

    cout
        << PRINT((my_func_reversed(2.0, 1)))
        << PRINT((my_func_reversed(1, 2.0)));
}


// Useful example: Constrain variadics based on last argument
/*
Scenario:
We like to add things to containers with a name, something of the form:
   add_item(const Item& item, const string& name, Container& c)
We can also construct an Item with a [awfully large] number of overloads, and 
we have convenenience overloads:
   add_item(${ITEM_CTOR_ARGS}, const string& name, Container& c)

C++11 / C++14 variadic templates / parameter packs do not permit function 
arguments to be specified after a parameter list; they must come first, not last.
To get them to come first, we can reverse the parameter list. Done.
*/

class Item {
public:
    Item(const string& name, int y, int extra = 10) {
        ostringstream os;
        os << "ctor1: " << name << " (" << y << ", " << extra << ")";
        value_ = os.str();
    }
    Item(int x, double y, const string& extra = "") {
        ostringstream os;
        os << "ctor2: " << x << ", " << y << " (" << extra << ")";
        value_ = os.str();
    }
    Item(const Item& other)
        : value_("ctor3: " + other.value_)
    { }
    Item(Item&&) = default;
    const string& value() const { return value_; }
private:
    string value_;
};

typedef pair<string, Item> Entry;
typedef vector<Entry> Container;
std::ostream& operator<<(std::ostream& os, const Container& c) {
    for (const auto& entry : c) {
        os << entry.first << " - " << entry.second.value() << endl;
    }
    return os;
}

void add_item(const Item& item, const string& name, Container& c) {
    cout << "add_item direct" << endl;
    c.push_back(Entry(name, item));
}

template<typename ... Args>
void add_item(Args&& ... args) {
    cout << "add_item variadic" << endl;
    static auto callable = VARIADIC_CALLABLE(add_item_reversed,);
    static auto reversed = stdcustom::make_callable_reversed(callable);
    reversed(std::forward<Args>(args)...);
}

template<typename ... RevArgs>
void add_item_reversed(Container& c, const string& name, RevArgs&&... revargs)
{
    cout << "add_item_reversed" << endl;
    // NOTE: Could use explicit factory in this case
    static auto ctor = VARIADIC_CALLABLE(Item,);
    static auto ctor_reversed = stdcustom::make_callable_reversed(ctor);
    auto item = ctor_reversed(std::forward<RevArgs>(revargs)...);
    add_item(item, name, c);
}

void useful_example() {
    Container c;
    add_item(Item("attribute", 12), "bob", c);
    add_item(Item(2, 2.5, "twelve"), "george", c);
    add_item(Item("nothing", 15), "something", c);
    add_item("nothing else", 16, "something", c);

    cout << PRINT(c);
}

int main() {
    cout << "--- simple ---" << endl;
    simple_example();
    cout << "--- advanced ---" << endl;
    advanced_example();
    cout << "--- useful ---" << endl;
    useful_example();

    return 0;
}
