#!/bin/bash
set -e -u

# Check timestamps before/after
[[ -d build/install ]] && (
    cd build/install
    find . | xargs touch -t 201701010500
    ls -l .
    )

# rm -rf build
./package_and_extract_drake.sh $DRAKE build/ build/install

ls -l build/install
