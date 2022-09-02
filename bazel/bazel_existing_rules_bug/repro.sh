#!/bin/bash
set -eux

scrub() {
    sed 's#'${PWD}'#${PWD}#g'
}

(
    bazel version 2>&1
    bazel build :workspace_lib 2>&1 | grep "DEBUG:"
) | scrub | tee ./repro.output.txt
