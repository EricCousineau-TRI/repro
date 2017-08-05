#!/bin/bash

# Expose environment from a Bazel target to test with things like MATLAB.

# NOTE: When trying to just execute a desired command, like `bash -i` or
# `matlab`, there are errors about `ioctl` and stuff like that, when using
# `bazel run`...
# (But how does `drake_visualizer` work???)
# According to GitHub issues, no one seems to care about resolving it?

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
    testonly = 1,
)
EOF

    mkdir -p $(dirname $script)
    # Generate environment and export it to a temporary file.
     # > /dev/null 2>&1 
    bazel run --spawn_strategy=standalone tmp:py_shell -- \
        bash -c "export -p > $script" \
            || { echo "Error for target: ${target}"; return 1;  }
    # Override PWD
    echo "declare -x PWD=$PWD" >> $script
}

script=$(cd $(dirname $BASH_SOURCE) && pwd)/tmp/bazel_env.sh
setup_target_env-main "$@" && {
    source $script;
    echo "[ Environment sourced for: ${target} ]"
}
