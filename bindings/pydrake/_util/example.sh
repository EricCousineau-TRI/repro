#!/bin/bash

source setup_target_env.sh --build bindings/pydrake bindings typebinding_test

cd $(dirname ${BASH_SOURCE})
python ../test/testTypeBinding.py
