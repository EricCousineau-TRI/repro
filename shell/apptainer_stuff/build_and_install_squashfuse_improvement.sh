#!/bin/bash
set -eux

# Based on
# https://github.com/apptainer/apptainer/blob/v1.1.3/INSTALL.md#installing-improved-performance-squashfuse_ll

sudo apt-get install \
     autoconf automake libtool pkg-config libfuse-dev zlib1g-dev

tmp_dir=/tmp/apptainer_provision/squashfuse
mkdir -p ${tmp_dir}
cd ${tmp_dir}

version=0.1.105
use_prs="70 77"
if [[ ! -f squashfuse-${version}.tar.gz ]]; then
    curl -L -O https://github.com/vasi/squashfuse/archive/${version}/squashfuse-${version}.tar.gz
fi
for PR in ${use_prs}; do
    curl -L -O https://github.com/vasi/squashfuse/pull/$PR.patch
done

rm -rf squashfuse-${version}
tar xzf squashfuse-${version}.tar.gz
cd squashfuse-${version}
for PR in ${use_prs}; do
    patch -p1 <../$PR.patch
done
./autogen.sh
FLAGS=-std=c99 ./configure --enable-multithreading
make squashfuse_ll -j
cp squashfuse_ll ~/.local/opt/apptainer/libexec/apptainer/bin
