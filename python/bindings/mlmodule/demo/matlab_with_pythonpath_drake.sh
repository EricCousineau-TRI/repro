#!/bin/bash
set -e -x -u

# Expose Python environment to MATLAB such that we can call pymodule without any
# install steps.

# Does so within Drake, and exposes the parts necessary to run MathematicalProgram.

orig_dir=$(pwd)
source_dir=$(cd $(dirname $BASH_SOURCE) && pwd)
pymodule_dir=$source_dir/../../pymodule

# Source Python environment for Drake.
cd ${DRAKE}/drake  # Just ${DRAKE} messes with Bazel trying to create a .tmp folder.
source ${pymodule_dir}/env/setup_target_env.sh \
    //drake/bindings:pydrake

# Ensure we can run the test script directly. Fail fast if this does not work.
python ${DRAKE}/drake/bindings/python/pydrake/test/testMathematicalProgram.py

# Start MATLAB, running startupProject MATLAB function.
cd ${orig_dir}
matlab -r startupProject
