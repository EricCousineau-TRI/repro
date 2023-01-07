#!/bin/bash
set -eux

# See https://apptainer.org/docs/user/1.1/quick_start.html#quick-installation-steps
# https://github.com/apptainer/apptainer/blob/v1.1.3/INSTALL.md

# From docs.
sudo apt-get install \
    build-essential \
    libseccomp-dev \
    pkg-config \
    uidmap \
    squashfs-tools \
    squashfuse \
    fuse2fs \
    fuse-overlayfs \
    fakeroot \
    cryptsetup \
    curl wget git

# TODO(eric.cousineau): Er, on Ubuntu 20.04 Focal, it says `fuse` was removed,
# and may create dependency issues w/ following:
#   sshfs ntfs-3g gvfs-fuse bindfs

# For not needing `sudo` - uidmap needed for --fakeroot to work
sudo apt-get install uidmap

# Prep for installation.
mkdir -p ~/.local/bin ~/.local/opt

export PATH=~/.local/bin:${PATH}

tmp_dir=~/devel/apptainer_provision
mkdir -p ${tmp_dir}

# Install go.
if ! which go ; then
    cd ${tmp_dir}
    if [[ ! -f ./go.tar.gz ]]; then
        GOVERSION=1.19.3
        OS=linux
        ARCH=amd64
        wget \
            https://dl.google.com/go/go${GOVERSION}.${OS}-${ARCH}.tar.gz \
            -O ./go.tar.gz
    fi
    mkdir -p ~/.local/opt/go
    tar -xzf ./go.tar.gz --strip-components=1 -C ~/.local/opt/go
    ln -sf ~/.local/opt/go/bin/go ~/.local/bin
fi

apptainer_versions="v1.1.4 v1.0.0 v0.1.1"
for apptainer_version in ${apptainer_versions}; do
    install_prefix=~/.local/opt/apptainer/${apptainer_version}
    apptainer_bin=${install_prefix}/bin/apptainer

    if [[ ! -f ${apptainer_bin} ]]; then
        cd ${tmp_dir}
        # Install apptainer.
        if [[ ! -d ./apptainer ]]; then
            git clone https://github.com/apptainer/apptainer
        fi
        cd apptainer
        git fetch
        git checkout ${apptainer_version}
        ./mconfig --without-suid -p ${install_prefix}
        make -C builddir -j
        make -C builddir install -j
        ln -sf ${apptainer_bin} ~/.local/bin/apptainer-${apptainer_version}
    fi

    if ! which apptainer ; then
        ln -sf ${apptainer_bin} ~/.local/bin/
    fi
done
