#!/bin/bash

source setup_target_env.sh

cd $(dirname ${BASH_SOURCE})
python ../pydrake/test/testTypeBinding.py
