#!/bin/bash
set -eux -o pipefail

cd $(dirname ${BASH_SOURCE})

binder_bin=~/devel/binder/build/source/binder
output_dir=/tmp/example_bind

rm -rf ${output_dir}

${binder_bin} \
    --root-module test_struct \
    --prefix ${output_dir}/ \
    --bind testers \
    example.cc \
    -- \
    -std=c++11 -DNDEBUG
