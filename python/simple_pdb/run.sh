#!/bin/bash

scrub() {
    sed \
        -e 's#'${PWD}/'#./#g' \
        -e 's#(Pdb) ##g' \
        -e 's#^> ##g'
}

input() {
    echo bt
    echo quit
}

input | python3 ./simple.py | scrub
