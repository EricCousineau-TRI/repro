#include <cassert>
#include <cstdio>
#include <dlfcn.h>
void modes() {
  const char* msg;
  switch (MODE) {
    case 0:
      msg = "Nominal lazy loading (won't work)";
      break;
    case 1:
      msg = "Include file to add to static symbol table.";
      break;
    case 2:
      msg = "Load initially using RTLD_GLOBAL";
      break;
    case 3:
      msg = "Reload using RTLD_NOLOAD | RTLD_GLOBAL";
      break;
    case 4:
      msg = "Reload (after calling first) using RTLD_NOLOAD | RTLD_GLOBAL";
      break;
  }
  printf("Mode %d: %s\n", MODE, msg);
}
#if MODE == 1
  // Add to static symbol table.
  #include "producer.h"
#endif
struct Library {
  const char* lib;
  void* handle;
  Library(const char* lib_in) {
    lib = lib_in;
    printf("dlopen: %s\n", lib);
#if MODE == 0 || MODE == 1 || MODE == 4
    handle = dlopen(lib, RTLD_LAZY);
#elif MODE == 2
    handle = dlopen(lib, RTLD_LAZY | RTLD_GLOBAL);
#elif MODE == 3
    handle = dlopen(lib, RTLD_LAZY);
    call();  // Check calling before re-loaded with GLOBAL
    dlopen(lib, RTLD_LAZY | RTLD_NOLOAD | RTLD_GLOBAL);
#endif
    assert(handle);
  }
  void call() {
    typedef void (*func_t)();
    func_t func = (func_t) dlsym(handle, "entry");
    func();
#if MODE == 4
    dlopen(lib, RTLD_LAZY | RTLD_NOLOAD | RTLD_GLOBAL);
    func();
    func = (func_t) dlsym(handle, "entry");
    func();
#endif
  }
  ~Library() {
    dlclose(handle);
    printf("dlclose: %s\n", lib);
  }
};
int main() {
  modes();
  {
    Library c1("./consumer_1.so");
    Library c2("./consumer_2.so");
    c1.call();
    c2.call();
  }
  printf("\n\n");
  return 0;
}
