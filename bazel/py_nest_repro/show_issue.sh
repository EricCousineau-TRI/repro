#!/bin/bash
set -e -u -x

bazel --bazelrc=/dev/null version

(
    echo "[ Working ]"
    cd sub_example
    bazel --bazelrc=/dev/null run //src/sub_example:usage_test
)

(
    echo "[ Not Working ]"
    bazel --bazelrc=/dev/null run :usage_test
)
