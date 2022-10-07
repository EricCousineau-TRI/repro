#!/bin/bash
set -eu

die() { echo "$@" >&2; exit 1; }

do-env-checks() {
    set -e
    false
    echo "should not reach"
}

subshell-test() { (
    do-env-checks
) }

positive-tests() {
    subshell-test || die "subshell failed"
}

positive-tests

echo "Done"
