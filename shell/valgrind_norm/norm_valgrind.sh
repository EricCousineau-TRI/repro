#!/bin/bash

bazel_cache="$HOME/.cache/bazel/_bazel_$USER"

cat | sed -r \
    -e "s#^==[0-9]+==#==XXXXX==#g" \
    -e "s#0x[0-9A-F]+#0xXXXXXX#g" \
    -e "s#${bazel_cache}/[0-9a-f]+/#/.../#g"
