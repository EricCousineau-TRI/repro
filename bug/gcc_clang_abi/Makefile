ifndef CXX_CLANG
	CXX_CLANG := clang++-3.9
endif
ifndef CXX_GCC
	CXX_GCC := g++-4.9
endif

CXX_FLAGS := -std=c++1y

.PHONY: all
all: main

# Compile library with clang
liblib.so: lib.cc lib.h
	$(CXX_CLANG) $(CXX_FLAGS) -shared -fPIC $< -o $@

main: main.cc liblib.so
	$(CXX_GCC) $(CXX_FLAGS) -c main.cc -o main.o
	$(CXX_GCC) $(CXX_FLAGS) main.o -o $@ -Wl,-rpath=$(PWD) -L$(PWD) -llib -lstdc++
