#!/bin/bash

source setup_target_env.sh bindings pydrake_type_binding_test

cd $(dirname ${BASH_SOURCE})
python ../pydrake/test/testTypeBinding.py
