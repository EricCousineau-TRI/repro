#!/bin/bash
set -eu

cd $(dirname $0)

(
    set -x
    env

    bazel --bazelrc=/dev/null run //:example_cc
    bazel --bazelrc=/dev/null run @example//:example_cc
    ldd bazel-bin/example_py.runfiles/example/example_lib_py.so | grep 'not found' || :

    bazel --bazelrc=/dev/null run //:example_py
    bazel --bazelrc=/dev/null run @example//:example_py || :
    ldd bazel-bin/external/example/example_py.runfiles/example/example_lib_py.so | grep 'not found'

    bazel --bazelrc=/dev/null run //:example_ldd
    bazel --bazelrc=/dev/null run @example//:example_ldd
) 2>&1 | \
    sed -e "s#$(bazel --bazelrc=/dev/null info workspace)#\${bazel_workspace}#g" \
        -e "s#/home/.*/_bazel_${USER}#\${bazel_cache}#g" \
        -e "s#${USER}#\${user}#g" | \
    tee repro.output.txt
