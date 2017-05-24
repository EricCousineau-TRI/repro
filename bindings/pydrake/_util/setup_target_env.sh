#!/bin/bash

# Expose Bazel's Python paths, using path info gleaned from `env_info.output.txt`.
#
# Usage:
# 
#    $ source setup_target_env.sh <PACKAGE> <IMPORTS> <TARGET>

set -x

cur=$(cd $(dirname $BASH_SOURCE) && pwd)

workspace=$(bazel info workspace)
# Use workaround for bazel <= 0.4.5
# @ref: https://github.com/bazelbuild/bazel/issues/2317#issuecomment-284725684
workspace_name=$($workspace/shell/bazel_workspace_name.py)

package=${1}
imports=${2}
target=${3}
runfiles=${workspace}/bazel-bin/${package}/${target}.runfiles

export PYTHONPATH=${runfiles}:${runfiles}/${workspace_name}/${imports}:${runfiles}/${workspace_name}:${PYTHONPATH}

set +x
echo "[ Exposed \${PYTHONPATH} for target: //${package}:${target} ]"
