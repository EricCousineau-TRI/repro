#!/bin/bash
set -eux

# Usage: build_image_docker.sh <name> <image>

name=${1}
image=${2}

sif=${name}.sif
sandbox=${sif}.sandbox

# See https://github.com/apptainer/singularity/issues/1537#issuecomment-557527638

# Build image.
apptainer -v build --fakeroot ${sif} docker-daemon://${image}

# Expand image into sandbox (so you can see files / make it easy to make
# writeable.).
apptainer build --fakeroot --sandbox ${sandbox} ${sif}
