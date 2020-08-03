#!/bin/bash
set -eux -o pipefail

cd $(dirname ${BASH_SOURCE})

drake_path=~/tmp/venv/drake
binder_bin=~/devel/binder/build/source/binder

${binder_bin} \
    --root-module sample \
    --prefix /tmp/binder \
    --bind drake \
    ./drake_headers.h \
    -- \
    --std=c++17 \
    -x c++ \
    -I${drake_path}/include \
    -I${drake_path}/include/fmt \
    -I/usr/include/eigen3
