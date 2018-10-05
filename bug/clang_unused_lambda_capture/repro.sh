#!/bin/bash
set -eux -o pipefail

compile() {
    clang++-6.0 -std=c++14 -Wall -Werror "$@"
    # g++ -std=c++14 -Wall -Werror "$@"
}

compile ./orig.cc -o ./orig
./orig
echo

compile ./test.cc -o ./test
./test
