#!/bin/bash
set -eux

# Usage: build_image.sh <base>

name=${1}

def=${name}.Apptainer
sif=${name}.sif
sandbox=${sif}.sandbox

# Meh. Dunno how to plumb in args.
configure-file() {
    in=${1}
    out=${2}
    sed -E \
        -e 's#@USER@#'${USER}'#g' \
        ${in} > ${out}
}

if [[ -f ${def}.in ]]; then
    configure-file ${def}.in ${def}
fi

# Build image.
apptainer build --fakeroot ${sif} ${def}

# TODO(eric.cousineau): Make sandbox optional.

# Expand image into sandbox (so you can see files / make it easy to make
# writeable.).
apptainer build --fakeroot --sandbox ${sandbox} ${sif}
