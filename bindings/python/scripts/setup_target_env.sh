#!/bin/bash

# Expose Bazel's Python paths, using path info gleaned from `env_info.output.txt`.
#
# Usage:
# 
#    $ source setup_target_env.sh <PACKAGE> <TARGET>

set -x

cur=$(cd $(dirname $BASH_SOURCE) && pwd)

workspace=$(bazel info workspace)
# Use workaround for bazel <= 0.4.5
# @ref: https://github.com/bazelbuild/bazel/issues/2317#issuecomment-284725684
workspace_name=$($cur/../../../shell/bazel_workspace_name.py)

package=${1}
import_path=${workspace_name}/${package}/python
target=${2}
runfiles=${workspace}/bazel-bin/${package}/${target}.runfiles

export PYTHONPATH=${runfiles}:${runfiles}/${import_path}:${runfiles}/${workspace_name}:${PYTHONPATH}

set +x
echo "[ Exposed \${PYTHONPATH} for target: //${package}:${target} ]"
