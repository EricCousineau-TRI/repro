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

check example_cc
check libexample_lib_cc.so
check libsecondary_direct_lib_cc.so
check libsecondary_indirect_lib_cc.so

exit 0
