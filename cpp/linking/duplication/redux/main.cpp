#include <cstdio>
#include <dlfcn.h>
#if MODE == 0
  // Add to static symbol table.
  #include "producer.h"
#endif
struct Library {
  const char* lib;
  void* handle;
  Library(const char* lib_in) {
    lib = lib_in;
    printf("dlopen: %s\n", lib);
#if MODE != 2
    handle = dlopen(lib, RTLD_LAZY);
#else
    handle = dlopen(lib, RTLD_LAZY | RTLD_GLOBAL);
#endif
    if (!handle) {
      fprintf(stderr, "%s\n", dlerror());
    }
    typedef void (*func_t)();
    func_t func = (func_t) dlsym(handle, "entry");
    func();
  }
  ~Library() {
    dlclose(handle);
    printf("dlclose: %s\n", lib);
  }
};
int main() {
  {
    Library c1("./consumer_1.so");
    Library c2("./consumer_2.so");
  }
  printf("\n\n");
  return 0;
}
