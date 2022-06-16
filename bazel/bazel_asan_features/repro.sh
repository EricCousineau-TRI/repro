#!/bin/bash
set -eux

scrub() {
    sed -E \
        -e "s#${PWD}#{proj}#g" \
        -e "s#0x[0-9a-f]+#0x{hex}#g" \
        -e "s#[0-9a-f\-]{6,}#{hex}#g" \
        -e "s#/.*?/execroot/#{execroot}/#g"
}

bazel_() {
    bazel --nohome_rc "$@"
}

bazel_ clean --expunge --async
(
    bazel version
    bazel_ build -j 1 -s //:example
    cat bazel-out/k8-opt/bin/example-2.params
) 2>&1 | scrub | tee repro.without.txt

bazel_ clean --expunge --async
(
    bazel version
    bazel_ build --features=asan -j 1 -s //:example
    cat bazel-out/k8-opt/bin/example-2.params
) 2>&1 | scrub | tee repro.with.txt

git --no-pager diff --no-index repro.*.txt | tee repro.diff
