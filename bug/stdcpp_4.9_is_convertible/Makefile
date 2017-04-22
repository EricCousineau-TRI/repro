.PHONY: all clean build

SHELL = bash

CXX_BAD = g++-4.9
CXXFLAGS_BAD = --std=c++1y -I/usr/include/c++/4.9/ -L/usr/lib/gcc/x86_64-linux-gnu/4.9/libstdc++.so

CXX = clang++
CXXFLAGS = --std=c++1y

all: run

build:
	mkdir -p build

build/test_bad: test.cc build
	$(CXX_BAD) $(CXXFLAGS_BAD) test.cc -o build/test_bad

build/test: test.cc build
	$(CXX) $(CXXFLAGS) test.cc -o build/test

run: build/test build/test_bad
	set -x; { { ./build/test; build/test_bad; }  2>&1; } | tee test.output.txt

clean:
	rm -rf build
