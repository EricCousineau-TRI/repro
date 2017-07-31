#!/bin/bash

# @see https://gcc.gnu.org/bugzilla/show_bug.cgi?id=57709

# Error
set -x
CXX=g++-4.9 make VERBOSE=1 -B

# Error
echo
CXX=g++-5 make VERBOSE=1 -B

# No error
echo
CXX=clang++ make VERBOSE=1 -B
