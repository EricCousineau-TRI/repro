#!/bin/bash

# Expose Bazel's Python paths, using path info gleaned from `env_info.output.txt`.
#
# Usage:
# 
#    $ source setup_target_env.sh <PACKAGE> <TARGET> [<WORKSPACE_NAME>]

set -x

workspace=$(bazel info workspace)
# @ref: https://github.com/bazelbuild/bazel/issues/2317#issuecomment-284725684
# Note: No dice. Still gives the wrong name
# With "drake", //drake/bindings:pydrake_*_test will place things under
# ${runfiles}/drake/drake/..., but using `workspace` or `execution_root` yeilds
# ${runfiles}/drake-distro/drake/....
workspace_name_default=$(basename $(bazel info execution_root))
workspace_name=${3-${workspace_name_default}}

package=${1}
import_path=${workspace_name}/${package}/python
target=${2}
runfiles=${workspace}/bazel-bin/${package}/${target}.runfiles

export PYTHONPATH=${runfiles}:${runfiles}/${import_path}:${runfiles}/${workspace_name}:${PYTHONPATH}

set +x
echo "[ Exposed \${PYTHONPATH} for target: //${package}:${target} ]"
