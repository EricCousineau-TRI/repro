#!/bin/bash
set -e -u -x

cd $(dirname $BASH_SOURCE)

if [[ ! -d tmp/env/bin/activate ]]; then
    mkdir -p tmp/env
    virtualenv --system-site-packages tmp/env
fi

set +e +u +x
source tmp/env/bin/activate
set -e -u -x

( cd python_pkg && pip install -I . )

python -c 'from python_pkg import some_func; print(some_func())'

(
    cd bazel_pkg
    bazel clean
    bazel run :python_pkg_test
    bazel test :python_pkg_test
)
