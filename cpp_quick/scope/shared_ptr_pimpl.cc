// Goal: Show that we can use PIMPL patterns with shared pointers.
// Try to hide as much as possible.

#include <iostream>
#include <memory>

using std::cout;
using std::endl;
using std::shared_ptr;
using std::make_shared;

// Forward declare, hide implementation
shared_ptr<void> get_resource();


int main() {
    shared_ptr<void> ptr = get_resource();
    return 0;
}

shared_ptr<void> get_resource() {
    class Resource {
     public:
        Resource() { cout << "ctor" << endl; }
        ~Resource() { cout << "dtor" << endl; }
    };
    return make_shared<Resource>();
}
