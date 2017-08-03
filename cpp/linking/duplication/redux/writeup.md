I am seeing if there are any minimally-hacky ways to be able combine shared objects at run-time (e.g., `dlopen`, loading a `pybind11` library), and ensure that their linkages do not create duplicate globals.

Presently, code like this:

`producer.h`

    #include <cstdio>
    void produce() {
        static int value = 0;
        ++value;
        fprintf("  produce: %d (%d)\n", value, &value);
    }

`consumer.cpp -> consumer_{1,2}.so`

    #include "producer.h"
    extern "C" void entry() {
        fprintf(CONSUMER "\n");
        produce();
    }

`main.cpp`

    #include  <dlfcn.h>
    #ifdef GOOD
      #include "producer.h"
      // Reference call.
      void unused() { produce(); }
    #endif
    void call(const char* lib) {
        void* handle = dlopen(lib, RTLD_NOW | RTLD_GLOBAL);
        typedef void (*func_t)();
        func_t func = (func_t) dlsym(handle, "entry");
        func();
        dlclose();
    }
    int main() {
        call("./consumer_1.so");
        call("./consumer_2.so");
        return 0;
    }

`Makefile`

    # From: https://stackoverflow.com/questions/8623657/multiple-instances-of-singleton-across-shared-libraries-on-linux
    FLAGS := $(CXXFLAGS) -fPIC -rdynamic
    all: main_bad main_good consumer_1.so consumer_2.so
    main_bad: main.cpp
        $(CXX) $(FLAGS) -o $@ $< -ldl
    main_good: main.cpp
        $(CXX) $(FLAGS) -DGOOD=1 -o $@ $< -ldl
    consumer_1.so: consumer.cpp
        $(CXX) $(FLAGS) -DCONSUMER=$@ -shared -o $@ $<
    consumer_2.so: consumer.cpp
        $(CXX) $(FLAGS) -DCONSUMER=$@ -shared -o $@ $<
