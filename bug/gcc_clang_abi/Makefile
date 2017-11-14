CXX_CLANG := clang
CXX_GCC := g++

CXX_FLAGS := -std=c++1y

.PHONY: all
all: main

# Compile library with clang
liblib.so: lib.cc lib.h
	$(CXX_CLANG) $(CXX_FLAGS) -shared -fPIC $< -o $@

main: main.cc liblib.so
	$(CXX_GCC) $(CXX_FLAGS) main.cc -o $@ -Wl,-rpath=$(PWD) -L$(PWD) -llib -lstdc++
