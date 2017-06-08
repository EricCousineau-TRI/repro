#!/bin/bash
set -e -u

show-timing() {
    prefix=${1-}
    { {
        echo "Prefix: ${prefix}"
        echo "[ With OpenMP ]"
        ( set -x; ./build/${prefix}with_openmp )
        echo "[ Without OpenMP ]"
        ( set -x; ./build/${prefix}without_openmp )
    } 2>&1 ; } | tee ${prefix}timing.txt 
}

make -j
show-timing ''
