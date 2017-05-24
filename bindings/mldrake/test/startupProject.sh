#!/bin/bash
set -e -x -u

source_dir=$(cd $(dirname $BASH_SOURCE) && pwd)
cd $source_dir

pydrake_dir=$source_dir/../../pydrake

# Source Python environment
source ${pydrake_dir}/_util/setup_target_env.sh bindings/pydrake bindings typebinding_test

# Ensure we can run the test script
python ${pydrake_dir}/test/testTypeBinding.py

# Start MATLAB, running startupProject MATLAB function.
matlab -r startupProject
