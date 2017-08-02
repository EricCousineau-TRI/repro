#include <iostream>
#include <dlfcn.h>

typedef void (*hello_t)();
using std::cout;
using std::cerr;
using std::endl;

void call_hello(const char* lib, const char* name) {

    // open the library
    void* handle = dlopen(lib, RTLD_LAZY);
    if (!handle) {
        cerr << "Cannot open library: " << dlerror() << '\n';
        return;
    }
    // load the symbol
    
    // reset errors
    dlerror();
    hello_t hello = (hello_t) dlsym(handle, name);
    const char *dlsym_error = dlerror();
    if (dlsym_error) {
        cerr << "Cannot load symbol 'hello': " << dlerror() << '\n';
        dlclose(handle);
        return;
    }
    hello(); // call plugin function hello
    dlclose(handle);
}

int main() {
    call_hello("./hello.so", "hello");
    call_hello("./hello2.so", "hello2");
    return 0;
}
