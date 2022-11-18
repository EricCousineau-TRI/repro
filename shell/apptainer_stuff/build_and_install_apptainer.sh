#!/bin/bash
set -eux

# Based on
# https://github.com/EricCousineau-TRI/repro/tree/b833082c/shell/apptainer_stuff#building-apptainer

# See https://apptainer.org/docs/user/1.0/quick_start.html#quick-installation-steps

# From docs.
sudo apt install \
   build-essential \
   libseccomp-dev \
   pkg-config \
   squashfs-tools \
   cryptsetup \
   curl wget git
# For not needing `sudo` - uidmap needed for --fakeroot to work
sudo apt install uidmap

# Prep for installation.
mkdir -p ~/.local/bin ~/.local/opt

export PATH=~/.local/bin:${PATH}

tmp_dir=/tmp/apptainer_provision
mkdir -p ${tmp_dir}
cd ${tmp_dir}

# Install go.
if ! which go ; then
    if [[ ! -f ./go.tar.gz ]]; then
        wget https://golang.org/dl/go1.18.linux-amd64.tar.gz -O ./go.tar.gz
    fi
    tar -xzf ./go.tar.gz -C ~/.local/opt
    ln -sf ~/.local/opt/go/bin/go ~/.local/bin
fi

if ! which apptainer ; then
    # Install apptainer.
    if [[ ! -d ./apptainer ]]; then
        git clone https://github.com/apptainer/apptainer
    fi
    cd apptainer
    git checkout v1.0.0
    ./mconfig --without-suid -p ~/.local/opt/apptainer
    make -C builddir -j 8
    make -C builddir install -j 8
    ln -sf ~/.local/opt/apptainer/bin/apptainer ~/.local/bin/
fi
