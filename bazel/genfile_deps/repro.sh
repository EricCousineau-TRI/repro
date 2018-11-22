#!/bin/bash
set -eux -o pipefail

cd $(dirname ${BASH_SOURCE})
die() { echo "$@" >&2; exit 1; }

echo "[ USE_WORKAROUND = False ]"
sed -i 's#^USE_WORKAROUND = .*$#USE_WORKAROUND = False#' tools/dumb_generator.bzl
bazel clean
bazel build //data:data && die "Should have died"

echo -e "\n\n\n"
echo "[ USE_WORKAROUND = False ]"
sed -i 's#^USE_WORKAROUND = .*$#USE_WORKAROUND = True#' tools/dumb_generator.bzl
bazel clean
bazel build //data:data || die "Should not have died"
