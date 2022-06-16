#!/bin/bash
set -eux

# Build with Bazel.
bazel clean --expunge --async

scrub() {
    sed -E \
        -e "s#${PWD}#{proj}#g" \
        -e "s#0x[0-9a-f]+#0x{hex}#g" \
        -e "s#[0-9a-f\-]{6,}#{hex}#g" \
        -e "s#/.*?/execroot/#{execroot}/#g"
}

(
    bazel build -j 1 -s //:example
) 2>&1 | scrub | tee repro.output.txt
