#!/bin/bash
set -eu

echo "pwd: $PWD"
echo "ld: $@"
/usr/bin/ld "$@"
