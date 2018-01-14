#!/bin/bash
set -e

ws=recurse_filegroup

indent() { sed 's#^#  #g'; }

run() {
    cmd=${1}
    echo "[ ${cmd} ]"
    echo "args:"
    bazel run :"${cmd}" 2> /dev/null | indent
    echo
    echo "runfiles:"
    (
        cd bazel-bin/${cmd}.runfiles/${ws}
        find . -type f -o -type l | indent
    )
    echo
}

run print
run print_execute
run print_recurse
