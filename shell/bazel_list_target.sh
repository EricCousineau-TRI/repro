#!/bin/bash
set -x -e -u

eecho() { echo "$@" >&2; }

for target in $@; do
    eecho "Target: $target"
    bazel query --output=location 'filter(".*'${target}'$", //...)'
done \
    | regex_sub.py '([/\w\d\-:]+):\d+:.*' '\1' - \
    | sort
