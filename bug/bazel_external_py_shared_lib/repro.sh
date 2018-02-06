#!/bin/bash
set -eu

# To run:
#  alias bash-isolate='env -i HOME=$HOME DISPLAY=$DISPLAY SHELL=$SHELL TERM=$TERM USER=$USER PATH=/usr/local/bin:/usr/bin:/bin bash --norc'
#  bash-isolate ./repro.sh

cd $(dirname $0)

(
    set -x
    env

    bazel --bazelrc=/dev/null test //:example_cc_direct //:example_cc_indirect
    bazel --bazelrc=/dev/null run //:example_ldd
    bazel --bazelrc=/dev/null test @example//:example_cc_direct @example//:example_cc_indirect
    bazel --bazelrc=/dev/null run @example//:example_ldd
) 2>&1 | \
    sed -e "s#$(bazel --bazelrc=/dev/null info workspace)#\${bazel_workspace}#g" \
        -e "s#/home/.*/_bazel_${USER}#\${bazel_cache}#g" \
        -e "s#${USER}#\${user}#g" | \
    tee repro.output.txt
