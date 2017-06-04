#!/bin/bash
set -e -x -u

# Expose Python environment to MATLAB such that we can call pymodule without any
# install steps.

source_dir=$(cd $(dirname $BASH_SOURCE) && pwd)
cd $source_dir

pymodule_dir=$source_dir/../../pymodule

# Source Python environment
source ${pymodule_dir}/_util/setup_target_env.sh \
    --build bindings/pymodule bindings typebinding_test

# Ensure we can run the test script directly. Fail fast if this does not work.
python ${pymodule_dir}/test/testTypeBinding.py

# Start MATLAB, running startupProject MATLAB function.
matlab -r startupProject
