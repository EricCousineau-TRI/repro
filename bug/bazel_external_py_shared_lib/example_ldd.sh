#!/bin/bash
set -x

src_dir=$(cd $(dirname $0) && pwd)
runfiles_dir=${PWD}
mkdir -p no_neighbor && cd no_neighbor

check() {
    local name=$1
    ldd ${src_dir}/${name} | grep "not found"
    ldd ${runfiles_dir}/${name} | grep "not found"
}

check example_cc_indirect
check example_cc_direct

exit 0
