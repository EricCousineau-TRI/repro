#!/bin/bash
set -eux

scrub() {
    sed -E \
        -e "s#${PWD}#{proj}#g" \
        -e "s#0x[0-9a-f]+#0x{hex}#g" \
        -e "s#[0-9a-f\-]{6,}#{hex}#g" \
        -e "s#/.*?/execroot/#{execroot}/#g"
}

cd $(dirname ${BASH_SOURCE})

bazel clean --expunge --async

{
    bazel build :directly_built :alias

    # Can run this binary, since it has runfiles.
    ./bazel-bin/directly_built

    # Cannot run this binary, since it does not exist.
    ! ./bazel-bin/alias

    # Cannot run this binary, since it's runfiles are not populated.
    ! ./bazel-bin/indirectly_built
} 2>&1 | scrub | tee ./output.txt
