#!/bin/bash

a=$1
b=$2

tmp=/tmp/norm_valgrind
mkdir -p $tmp/{a,b}
a_norm=$tmp/a/$(basename $a)
b_norm=$tmp/b/$(basename $b)

cur=$(cd $(dirname $BASH_SOURCE) && pwd)
$cur/norm_valgrind.sh < $a > $a_norm
$cur/norm_valgrind.sh < $b > $b_norm

git diff --no-index $a_norm $b_norm
