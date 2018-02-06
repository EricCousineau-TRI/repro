#!/bin/bash
set -eux

src_dir=$(cd $(dirname $0) && pwd)
runfiles_dir=${PWD}
mkdir -p no_neighbor && cd no_neighbor

ldd ${src_dir}/libexample_lib_cc.so | grep "not found" || :
ldd ${runfiles_dir}/libexample_lib_cc.so | grep "not found" || :
ldd ${src_dir}/example_lib_py.so | grep "not found" || :
ldd ${runfiles_dir}/example_lib_py.so | grep "not found" || :
