#!/bin/bash
set -eu -o pipefail

die() { echo "$@" >&2; exit 1; }

do-env-checks() {
    false
    echo "should not reach"
}

subshell-test() { (
    do-env-checks
) }

# WHY DOES THIS MAKE THINGS BREAK?!!!!
positive-tests() {
    subshell-test || die "subshell failed"
}

positive-tests

echo "Done"
