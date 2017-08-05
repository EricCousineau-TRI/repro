#!/bin/bash

# Expose environment from a Bazel target to test with things like MATLAB.

# First, synthesize a fake script to leverage the original environment.
setup_target_env-main() {
    target=$1

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

    # Generate environment and export it to a temporary file.
    bazel run --spawn_strategy=standalone tmp:py_shell -- \
        bash -c "export -p > $script" \
        > /dev/null 2>&1 || { echo "Error for target: ${target}"; return 1;  }
    # Override PWD
    echo "declare -x PWD=$PWD" >> $script
}

script=$(cd $(dirname $BASH_SOURCE) && pwd)/tmp/bazel_env.sh
setup_target_env-main "$@" && {
    source $script;
    echo "[ Environment sourced for: ${target} ]"
}
