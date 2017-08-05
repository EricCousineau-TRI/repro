#!/bin/bash

# First, synthesize a fake script to leverage the original environment.
setup_target_env-main() {
    local target=$1

    # Use //tools:py_shell as the source file, and add the target such that
    # any necessary dependencies are pulled in.
    cat > BUILD <<EOF
py_binary(
    name = "py_shell",
    srcs = ["//tools:py_shell"],
    deps = [
        "${target}",
    ],
)
EOF

    local prefix=/tmp/bazel_env
    mkdir -p $prefix
    local env_dir=$(mktemp -d -p ${prefix})
    local script=${env_dir}/bazel_env.sh

    # Generate environment and export it to a temporary file.
    bazel run --spawn_strategy=standalone :py_shell -- \
        bash -c "export -p > $script"
    echo $script

    # Source environment.
    echo "[ Environment sourced for: $target ]"
    . $script
    cd .
}

setup_target_env-main "$@"
