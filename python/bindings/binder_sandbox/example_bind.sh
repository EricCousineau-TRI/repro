#!/bin/bash
set -eux -o pipefail

cd $(dirname ${BASH_SOURCE})

binder_bin=~/devel/binder/build/source/binder

${binder_bin} \
    --root-module test_struct \
    --prefix /tmp/example_bind/ \
    --bind testers \
    example.cc \
    -- \
    -std=c++11 -DNDEBUG
