#!/bin/bash
set -x

bazel_run() {
    bazel run -c dbg --run_under='valgrind --tool=memcheck' "$@"
}
target=":conservative_resize_issue"

echo "[ Good ]"
bazel_run $target

echo -e "\n\n\n"
echo "[ Bad ]"
bazel_run --copt='-DUSE_BAD' $target
