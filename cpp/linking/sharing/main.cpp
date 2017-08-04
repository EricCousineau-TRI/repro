#include <cassert>
#include <cstdio>
#include <dlfcn.h>

#include "producer.h"

// Test inter-shared object function sharing
// (but the shared functions themselves will not have stored memory.)
// TTBOMK, no leaks reported from valgrind:
//   valgrind --tool=memcheck ./main_0

struct Library {
  const char* lib;
  void* handle;
  Library(const char* lib_in) {
    lib = lib_in;
    printf("dlopen: %s\n", lib);
    handle = dlopen(lib, RTLD_LAZY); // | RTLD_GLOBAL);
    assert(handle);
  }
  funcs_t call(funcs_t in) {
    typedef funcs_t (*entry_t)(funcs_t);
    entry_t entry = (entry_t) dlsym(handle, "entry");
    return entry(in);
  }
  ~Library() {
    dlclose(handle);
    printf("dlclose: %s\n", lib);
  }
};

void tmp() {
  printf("tmp: main\n");
}
static void tmp_static() {
  printf("tmp_static: main\n");
}

int main() {
  funcs_t mine(&tmp, &tmp_static);
  {
    Library c1("./consumer_1.so");
    Library c2("./consumer_2.so");
    funcs_t c1_funcs = c1.call(mine);
    funcs_t c2_funcs = c2.call(c1_funcs);
    printf("[ main ]\n");
    c2_funcs();
  }
  printf("\n\n");
  return 0;
}
