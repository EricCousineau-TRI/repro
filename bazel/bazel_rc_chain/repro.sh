#!/bin/bash
set -x -e -u

bazel run :script
bazel run --define=HAS_1=ON --define=HAS_2=ON :script
bazel --bazelrc=./bazel_1.rc run :script
# Does not work.
bazel --bazelrc=./bazel_1.rc --bazelrc=./bazel_2.rc run :script
