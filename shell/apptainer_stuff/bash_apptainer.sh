#!/bin/bash

# To use this, source in your session. For use everywhere, place in your
# ~/.bash_aliases, e.g.
#   source /path/to/apptainer_stuff/bash_apptainer.sh

_apptainer_stuff=$(cd $(dirname ${BASH_SOURCE}) && pwd)

# REPLACE WITH YOUR USER NAME
_expected_user="<user>"

# Update some config based on apptainer usage.
apptainer-setup() {
    if [[ -v APPTAINER_NAME && ! -v _APPTAINER_WRAP && -d "/home/${_expected_user}" ]]; then
        export _APPTAINER_WRAP=1
        if [[ "${HOME}" == "/root" ]]; then
            export _APPTAINER_ROOT=1
            export HOME="/home/${_expected_user}"
        fi
        if echo ${PWD} | grep -E "^/root" 2>&1 > /dev/null; then
            _new_pwd=$(echo ${PWD} | sed -E 's#^/root#/home/'${_expected_user}'#g')
            cd ${_new_pwd}
        fi
    fi
    if [[ -v _APPTAINER_ROOT ]]; then
        export PS1="(appt root) ${PS1}"
    else
        export PS1="(appt) ${PS1}"
    fi
}

# Run setup.
apptainer-setup

apptainer-exec() {
    # Minimal containment. Allows access to home directory etc.
    # Use --fakeroot so that you can `apt install` stuff.
    # Image includes stubbed out `sudo` so scripts won't fail.

    # WARNING: You may need to add / adjust --bind to leverage any secondary
    # drives you may have.
    # For me, I add `--bind /mnt`.
    image=${1}
    shift
    apptainer --silent exec \
        --nv --writable \
        --fakeroot \
        --bind /mnt \
        "$@" \
        ${image} \
        bash
}

# Specific executions.

apptainer-jammy() {
    apptainer-exec ${_apptainer_stuff}/jammy.sif.sandbox
}

apptainer-jammy-ros() {
    apptainer-exec ${_apptainer_stuff}/jammy-ros-humble-desktop-ext.sif.sandbox
}

# Extra.

# Custom stuff to run in container for simple dev experience.
# This is what I (Eric) use.
apptainer-container-install() {
    apt install bash-completion git git-gui gitk wget xclip tmux
    apt install python-is-python3
    apt install nvidia-utils-470
}
