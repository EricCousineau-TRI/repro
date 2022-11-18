#!/bin/bash

# To use this, source in your session. For use everywhere, place in your
# ~/.bash_aliases, e.g.
#   source /path/to/apptainer_stuff/bash_apptainer.sh

_apptainer_stuff=$(cd $(dirname ${BASH_SOURCE}) && pwd)

# Update some config based on apptainer usage.
_apptainer-check() {
    # REPLACE WITH YOUR USER NAME
    expected_user="<user>"

    if [[ -v APPTAINER_NAME && -d "/home/${expected_user}" ]]; then
        if [[ "${HOME}" == "/root" ]]; then
            export _APPTAINER_ROOT=1
            export HOME="/home/${expected_user}"
        fi
        if echo ${PWD} | grep -E "^/root" 2>&1 > /dev/null; then
            _new_pwd=$(echo ${PWD} | sed -E 's#^/root#/home/'${expected_user}'#g')
            cd ${_new_pwd}
        fi
        if [[ -v _APPTAINER_ROOT ]]; then
            export PS1="(appt root) ${PS1}"
        else
            export PS1="(appt) ${PS1}"
        fi
    fi
}

_apptainer-check

apptainer-jammy() {
    # Minimal containment. Allows access to home directory etc.
    # Use --fakeroot so that you can `apt install` stuff.
    # Image includes stubbed out `sudo` so scripts won't fail.

    # WARNING: You may need to add / adjust --bind to leverage any secondary
    # drives you may have.
    # For me, I add `--bind /mnt`.
    apptainer --silent exec \
        --nv --writable \
        --fakeroot \
        --bind /mnt \
        "$@" \
        ${_apptainer_stuff}/jammy.sif.sandbox \
        bash
}

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
