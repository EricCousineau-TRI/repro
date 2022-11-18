#!/bin/bash

# To use this, source in your session. For use everywhere, place in your
# ~/.bash_aliases, e.g.
#   source /path/to/apptainer_stuff/bash_extra.sh

_apptainer_stuff=$(cd $(dirname ${BASH_SOURCE}) && pwd)

apptainer-ros-jammy() {
    image_ext=jammy-ros-humble-desktop-ext
    apptainer --silent exec \
        --nv --writable \
        --fakeroot \
        --bind /mnt \
        "$@" \
        ${_apptainer_stuff}/${image_ext}.sif.sandbox \
        bash
}
