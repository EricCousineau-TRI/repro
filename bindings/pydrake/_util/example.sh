#!/bin/bash

source setup_target_env.sh bindings/pydrake bindings typebinding_test

cd $(dirname ${BASH_SOURCE})
python ../pydrake/test/testTypeBinding.py
