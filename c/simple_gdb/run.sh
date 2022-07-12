#!/bin/bash
set -eu

gcc -g main.c -o ./main
gdb --batch -n ./main \
    -ex "run" \
    -ex "bt" \
    -ex "f 1"
