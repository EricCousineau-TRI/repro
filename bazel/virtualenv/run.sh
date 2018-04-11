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

( cd pkg && pip install . )

python -c 'from some_package import some_func; print(some_func())'
