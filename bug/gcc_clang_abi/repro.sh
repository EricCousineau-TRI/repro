#!/bin/bash
set -e -u

# @ref https://stackoverflow.com/questions/36159238/linking-problems-due-to-symbols-with-abicxx11

should_fail() { echo "Should have failed!" >&2; exit 1; }

show_obj() {
    echo "liblib.so:"
    nm -C liblib.so | grep get_value
    echo "main.o:"
    nm -C main.o | grep get_value
}

echo "[ Will fail ]"
make -B CXX_GCC=g++-4.9 && should_fail
show_obj

echo
echo "[ Won't fail ]"
make -B CXX_GCC=g++-5
show_obj
