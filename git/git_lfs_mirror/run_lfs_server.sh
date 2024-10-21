#!/bin/bash
set -eux

cd $(dirname ${BASH_SOURCE})

export LFS_ADMINUSER=test
export LFS_ADMINPASS=test
build/lfs-test-server/lfs-test-server -port=8080 -path=${PWD}/lfs-content
