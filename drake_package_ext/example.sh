#!/bin/bash
set -e -u -x

# rm -rf build
./package_and_extract_drake.sh $DRAKE build/ build/install
