#!/bin/bash
set -eux -o pipefail

compile() {
    clang++-6.0 -Wall -Werror "$@"
}

compile ./orig.cc -o ./orig
./orig
echo

compile ./test.cc -o ./test
./test
