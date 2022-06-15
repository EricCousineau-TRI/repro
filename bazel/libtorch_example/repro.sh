#!/bin/bash
set -eux

bazel build //:libtorch_test

scrub() {
    sed -E -e "s#${PWD}#{proj}#g" -e "s#0x[0-9a-f]+#0x{hex}#g"
}

(
    ! bazel-bin/libtorch_test
    echo
    ldd bazel-libtorch_example/external/libtorch/lib/{*.so,*.so.*}
    echo
    ldd bazel-bin/libtorch_test
) 2>&1 | scrub | tee repro.output.txt
