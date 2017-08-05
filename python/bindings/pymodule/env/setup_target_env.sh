#!/bin/bash

# First, synthesize a fake script to leverage the original environment.
setup_target_env-main() {
    local target=$1

    # Use //tools:py_shell as the source file, and add the target such that
    # any necessary dependencies are pulled in.
    mkdir -p tmp
    cat > tmp/BUILD <<EOF
# NOTE: This is a temporary file. Do not version control!
py_binary(
    name = "py_shell",
    srcs = ["//tools:py_shell"],
    deps = [
        "${target}",
    ],
)
EOF

    echo $PWD
    local script=$PWD/tmp/bazel_env.sh

    # Generate environment and export it to a temporary file.
    bazel run --spawn_strategy=standalone tmp:py_shell -- \
        bash -c "export -p > $script"
    echo $script

    # Source environment.
    echo "[ Environment sourced for: $target ]"
    set -x
    source $script
    set +x
    cd .
}

setup_target_env-main "$@"
