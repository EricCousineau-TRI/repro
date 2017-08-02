#include <iostream>
#include <stdexcept>
#include <dlfcn.h>
#include "singleton.h"

typedef void (*hello_t)();
using std::cout;
using std::cerr;
using std::endl;

void call_hello(const char* lib, const char* name) {
    std::cout << singleton::pInstance << std::endl;
    // Without this first call, each library will have its own `count`.
    // With it, the count continues on.
    std::cout << &singleton::instance() << std::endl;
    // open the library
    void* handle = dlopen(lib, RTLD_LAZY);
    if (!handle) {
        throw std::runtime_error("Cannot open lib");
        return;
    }
    // load the symbol
    
    // reset errors
    dlerror();
    hello_t hello = (hello_t) dlsym(handle, name);
    const char *dlsym_error = dlerror();
    if (dlsym_error) {
        dlclose(handle);
        throw std::runtime_error("Cannot load symbol 'hello'");
    }
    hello(); // call plugin function hello
    dlclose(handle);
}

int main() {
    call_hello("./hello.so", "hello");
    call_hello("./hello2.so", "hello2");
    return 0;
}
