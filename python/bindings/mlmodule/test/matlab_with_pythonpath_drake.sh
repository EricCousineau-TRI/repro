#!/bin/bash
set -e -x -u

# Expose Python environment to MATLAB such that we can call pymodule without any
# install steps.

# Does so within Drake, and exposes the parts necessary to run MathematicalProgram.

source_dir=$(cd $(dirname $BASH_SOURCE) && pwd)
pymodule_dir=$source_dir/../../pymodule

# Source Python environment for Drake.
cd ${DRAKE}
source ${pymodule_dir}/_util/setup_target_env.sh \
    --build drake/bindings drake/bindings/python pydrake_mathematical_program_test

# Ensure we can run the test script directly. Fail fast if this does not work.
python ${DRAKE}/drake/bindings/python/pydrake/test/testMathematicalProgram.py

# Start MATLAB, running startupProject MATLAB function.
cd ${source_dir}
matlab -r startupProject
