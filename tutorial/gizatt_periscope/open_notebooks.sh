#!/bin/bash
set -eu

cd $(dirname ${BASH_SOURCE})

if [[ ! -d build ]]; then
    echo "Run ./build.sh first"
    exit 1;
fi

set +e +u
source ${PWD}/build/venv/bin/activate
set -eu

drake_install=${PWD}/build/drake
underactuated=${PWD}/build/underactuated
export PYTHONPATH=${underactuated}/src:${drake_install}/lib/python2.7/site-packages:${PYTHONPATH}

jupyter notebook
