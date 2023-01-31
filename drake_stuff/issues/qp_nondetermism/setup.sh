#!/bin/bash

# Simple bash wrapper for venv setup.
#
# Either source this, or use it as a prefix:
#
#   source ./setup.sh
#   ./my_program
#
# or
#
#   ./setup.sh ./my_program

_cur_dir=$(cd $(dirname ${BASH_SOURCE}) && pwd)
_venv_dir=${_cur_dir}/venv

_setup_venv() { (
    set -eu
    cd ${_cur_dir}

    if [[ -d "${_venv_dir}" ]]; then
        return 0
    fi

    set -x
    python3 -m venv ${_venv_dir}
    cd ${_venv_dir}
    ./bin/pip install -U pip
    ./bin/pip install -r ${_cur_dir}/requirements.txt
    ./bin/pip freeze > ${_cur_dir}/requirements.freeze.txt
) }

_setup_venv && source ${_venv_dir}/bin/activate

if [[ ${0} == ${BASH_SOURCE} ]]; then
    # This was executed, *not* sourced. Run arguments directly.
    exec "$@"
fi
