#!/bin/bash

# Expose Bazel's Python paths, using path info gleaned from `env_info.output.txt`.
#
# Usage:
# 
#    $ source setup_target_env.sh [<TARGET>] [<PACKAGE>]

package_default=bindings
target_default=pydrake_type_binding_test

workspace=$(bazel info workspace)
workspace_name=$(basename $workspace)

package=${2-${package_default}}
import_path=${workspace_name}/${package}/python
target=${1-${target_default}}
runfiles=${workspace}/bazel-bin/${package}/${target}.runfiles

export PYTHONPATH=${runfiles}:${runfiles}/${import_path}:${runfiles}/${workspace_name}:${PYTHONPATH}
echo "[ Exposed \${PYTHONPATH} for target: //${package}:${target} ]"
