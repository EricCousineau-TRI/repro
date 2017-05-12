#!/bin/bash
set -e -x -u

cur=$(cd $(dirname $BASH_SOURCE) && pwd)

sha_upstream=dec5d14
sha_downstream=5c549ab

$cur/../gen_patch.sh $cur/mosek_solver_test-{$sha_upstream,$sha_downstream}.output.txt > $cur/mosek_solver_test.patch
