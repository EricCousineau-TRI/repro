#!/bin/bash
set -eu

# To run:
#  alias bash-isolate='env -i HOME=$HOME DISPLAY=$DISPLAY SHELL=$SHELL TERM=$TERM USER=$USER PATH=/usr/local/bin:/usr/bin:/bin bash --norc'
#  bash-isolate ./repro.sh

cd $(dirname $0)

bazel="bazel --bazelrc=/dev/null"

strip() {
    sed -e "s#$(${bazel} info workspace)#\${bazel_workspace}#g" \
        -e "s#/home/.*/_bazel_${USER}#\${bazel_cache}#g" \
        -e "s#${USER}#\${user}#g"
}

(
    set -x
    env | strip

    {
        ${bazel} test //:example_cc_direct //:example_cc_indirect
        ${bazel} run //:example_ldd
    } 2>&1 | strip | tee /tmp/local.txt
    {
        ${bazel} test @example//:example_cc_direct @example//:example_cc_indirect
        ${bazel} run @example//:example_ldd
    } 2>&1 | strip | tee /tmp/external.txt

) 2>&1 | strip | tee repro.output.txt

git diff --no-index /tmp/local.txt /tmp/external.txt > /tmp/patch.diff
