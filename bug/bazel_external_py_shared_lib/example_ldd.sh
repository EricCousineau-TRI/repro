#!/bin/bash
set -eux

bin_dir=$(cd $(dirname $0) && pwd)

mkdir -p no_neighbor && cd no_neighbor

pwd
ldd ${bin_dir}/libexample_lib_cc.so | grep "not found" || :
ldd ${bin_dir}/example_lib_py.so | grep "not found" || :
