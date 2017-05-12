#!/bin/bash

cur=$(cd $(dirname $BASH_SOURCE) && pwd)
{
    bazel run --run_under='valgrind --tool=memcheck' :mosek_solver_test 2>&1 ;
} | tee $cur/mosek_solver_test-$(git ref).output.txt
