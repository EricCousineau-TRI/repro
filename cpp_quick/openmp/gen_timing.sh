#!/bin/bash
set -e -u

show-timing() {
    prefix=${1-}
    shift
    { {
        echo "Prefix: ${prefix}"
        echo "  Args: $@"
        echo "[ With OpenMP ]"
        ( set -x; ./build/${prefix}with_openmp "$@" )
        echo "[ Without OpenMP ]"
        ( set -x; ./build/${prefix}without_openmp "$@" )
    } 2>&1 ; } | tee ${prefix}timing.txt 
}

make -j
show-timing ''
show-timing '' --no-sleep
