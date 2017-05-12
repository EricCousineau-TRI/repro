#!/bin/bash

cur=$(cd $(dirname $BASH_SOURCE) && pwd)
{
    bazel run --run_under='valgrind --tool=memcheck --track-origins=yes --leak-check=full --show-leak-kinds=all' :mosek_solver_test 2>&1 ;
} | tee $cur/mosek_solver_test-$(git ref).output.txt
