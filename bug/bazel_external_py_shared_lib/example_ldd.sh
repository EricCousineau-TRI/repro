#!/bin/bash
set -x

src_dir=$(cd $(dirname $0) && pwd)
runfiles_dir=${PWD}
mkdir -p no_neighbor && cd no_neighbor

check-lib() {
    local name=$1
    ldd ${src_dir}/${name} | grep "not found"
    ldd ${runfiles_dir}/${name} | grep "not found"
}

check-lib libexample_lib_cc.so
check-lib example_lib_py.so
check-lib libsecond_lib_cc.so

exit 0
