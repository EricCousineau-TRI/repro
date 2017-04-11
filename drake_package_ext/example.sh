#!/bin/bash
set -e -u

echo "[ Before ]"
# Check timestamps before/after
[[ -d build ]] && (
    cd build
    find . | xargs touch -h -t 201701010500
    ls -l install
    )

echo "[ During ]"
# rm -rf build
./package_and_extract_drake.sh $DRAKE build/ build/install

echo "[ After ]"
ls -l build/install
