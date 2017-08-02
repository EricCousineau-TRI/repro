#include <iostream>
#include <dlfcn.h>
#include "singleton.h"

int main() {
    using std::cout;
    using std::cerr;
    using std::endl;

    singleton::instance().num = 100; // call singleton
    cout << "singleton.num in main : " << singleton::instance().num << endl;// call singleton

    // open the library
    void* handle = dlopen("./hello.so", RTLD_LAZY);

    if (!handle) {
        cerr << "Cannot open library: " << dlerror() << '\n';
        return 1;
    }

    // load the symbol
    typedef void (*hello_t)();

    // reset errors
    dlerror();
    hello_t hello = (hello_t) dlsym(handle, "hello");
    const char *dlsym_error = dlerror();
    if (dlsym_error) {
        cerr << "Cannot load symbol 'hello': " << dlerror() << '\n';
        dlclose(handle);
        return 1;
    }

    hello(); // call plugin function hello

    cout << "singleton.num in main : " << singleton::instance().num << endl;// call singleton
    dlclose(handle);
}
