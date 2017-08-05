#!/bin/bash
set -e -u

# Expose Python environment to MATLAB such that we can call pymodule without any
# install steps.

source_dir=$(cd $(dirname $BASH_SOURCE) && pwd)
cd $source_dir

pymodule_dir=$source_dir/../../pymodule

# Source Bazel python environment
source ${pymodule_dir}/env/setup_target_env.sh \
    //python/bindings/pymodule/sub:inherit_check_test

# Ensure we can run the test script directly. Fail fast if this does not work.
python ${pymodule_dir}/sub/test/inherit_check_test.py

# Start MATLAB, running startupProject MATLAB function.
matlab -r startupProject
