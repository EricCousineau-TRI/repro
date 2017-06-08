#!/bin/bash
set -e -u

show-timing() {
    prefix=${1-}
    shift
    {
        echo "Prefix: ${prefix}"
        echo "  Args: $@"
        echo "[ With OpenMP ]"
        ( set -x; ./build/${CXX}/${prefix}with_openmp "$@" )
        echo "[ Without OpenMP ]"
        ( set -x; ./build/${CXX}/${prefix}without_openmp "$@" )
    } 2>&1
}

compile-run() {
    echo "CXX: ${CXX}"
    make -j
    prefix=
    show-timing "${prefix}"
    show-timing "${prefix}" --no-sleep
    show-timing "${prefix}" --no-pragma
    show-timing "${prefix}" --no-pragma --no-sleep
}

{
    CXX=g++ compile-run
    # CXX=clang++ compile-run  # Does not work...
} | tee timing.txt
