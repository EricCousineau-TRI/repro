#!/bin/bash

show-timing() {
    prefix=${1-}
    echo "Prefix: ${prefix}"
    echo "[ With OpenMP ]"
    ( set -x; ./build/${prefix}with_openmp )
    echo "[ Without OpenMP ]"
    ( set -x; ./build/${prefix}without_openmp )
}

make -j
show-timing ''
