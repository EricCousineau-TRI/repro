#!/bin/bash

# To use this, source in your session. For use everywhere, place in your
# ~/.bash_aliases, e.g.
#   source /path/to/apptainer_stuff/bash_extra.sh

_apptainer_stuff=$(cd $(dirname ${BASH_SOURCE}) && pwd)

apptainer-ros-jammy() {
    apptainer --silent exec \
        --nv --writable "$@" \
        ${_apptainer_stuff}/jammy-ros-humble-desktop.sif.sandbox \
        bash
}
