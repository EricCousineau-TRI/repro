#!/bin/bash
set -eux -o pipefail

cd $(dirname ${BASH_SOURCE})

drake_path=~/venv/drake
binder_bin=~/devel/binder/build/llvm-6.0.1/build_6.0.1.linux.eacousineau-workstation.release/bin/binder

${binder_bin} \
    --root-module sample \
    --prefix /tmp/binder/ \
    --annotate-includes \
    --config ./drake_bind_config.cfg \
    ./drake_headers.h \
    -- \
    --std=c++17 \
    -x c++ \
    -I${drake_path}/include \
    -I${drake_path}/include/fmt \
    -I/usr/include/eigen3
