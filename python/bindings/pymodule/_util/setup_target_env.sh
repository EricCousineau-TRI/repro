#!/bin/bash

# Expose Bazel's Python paths, using path info gleaned from `env_info.output.txt`.
# Optionally build target beforehand if needed.
#
# Usage:
# 
#    $ source setup_target_env.sh [--build] <PACKAGE> <IMPORTS> <TARGET>

# https://unix.stackexchange.com/questions/219314/best-way-to-make-variables-local-in-a-sourced-bash-script
setup_target_env_impl() {
    set -x
    local source_dir=$(cd $(dirname $BASH_SOURCE) && pwd)

    local build=
    while [[ $# -gt 0 ]]; do
        case $1 in
            --build)
                build=1
                shift;;
            *)
                break;;
        esac
    done

    local workspace=$(bazel info workspace)
    # Use workaround for bazel <= 0.4.5
    # @ref: https://github.com/bazelbuild/bazel/issues/2317#issuecomment-284725684
    local workspace_name=$($(cd ${source_dir} && bazel info workspace)/shell/bazel_workspace_name.py)

    local package=${1}
    local imports=${2}
    local target=${3}
    local runfiles=${workspace}/bazel-bin/${package}/${target}.runfiles

    if [[ -n ${build} ]]; then
        bazel build //${package}:${target}
    fi

    export PYTHONPATH=${runfiles}:${runfiles}/${workspace_name}/${imports}:${runfiles}/${workspace_name}:${PYTHONPATH}

    set +x
    echo "[ Exposed \${PYTHONPATH} for target: //${package}:${target} ]"
}

setup_target_env_impl "$@"
