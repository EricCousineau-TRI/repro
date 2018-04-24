#!/bin/bash
set -ux

bazel version

rm -rf build
mkdir build && cd build

cat > WORKSPACE <<EOF
workspace(name = "example")
EOF

cat > BUILD.bazel <<EOF
cc_binary(
    name = "libexample_lib.so",
    srcs = [
        "example_shared.cc",
        "example_shared.h",
    ],
    linkshared = 1,
    linkstatic = 1,
)

cc_import(
    name = "example_lib",
    hdrs = ["example_shared.h"],
    shared_library = "libexample_lib.so",
)

cc_test(
    name = "example_cc",
    srcs = [
        "example.cc",
    ],
    deps = [":example_lib"],
)
EOF

cat > example_shared.h <<EOF
#pragma once

int func();
EOF

cat > example_shared.cc <<EOF
#include "example_shared.h"

int func() {
  return 10;
}
EOF

cat > example.cc <<EOF
#include "example_shared.h"

int main() {
  if (func() == 10)
    return 0;
  else
    return 1;
}
EOF

bazel run //:example_cc
ldd bazel-bin/example_cc | grep 'not found'  # Good.
ldd bazel-bin/example_cc.runfiles/example/example_cc | grep 'not found'  # Good.

bazel run @example//:example_cc
ldd bazel-bin/external/example/example_cc | grep 'not found'  # Good.
ldd bazel-bin/external/example/example_cc.runfiles/example/example_cc | grep 'not found'  # BAD
