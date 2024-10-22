#!/bin/bash
set -eux

# should install go:
cd $(dirname ${BASH_SOURCE})

mkdir -p build
cd build
if [[ ! -d lfs-test-server ]]; then
    git clone https://github.com/git-lfs/lfs-test-server
fi
cd lfs-test-server
go build

ls ./lfs-test-server
