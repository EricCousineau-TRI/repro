#!/bin/bash

_script_dir=$(cd $(dirname ${BASH_SOURCE}) && pwd)
_venv_dir=${_script_dir}/venv

_venv-setup() { (
    set +eux

    files-equal() {
        cmp "$@" > /dev/null 2>&1
    }

    requirements_in=${_script_dir}/requirements.txt
    requirements_used=${_venv_dir}/requirements_used.txt
    if ! files-equal ${requirements_in} ${requirements_used}; then
        rm -rf ${_venv_dir}
        python3 -m virtualenv -p python3 ${_venv_dir}
        ${_venv_dir}/bin/pip install -r ${requirements_in}
        cp ${requirements_in} ${requirements_used}
    fi
) }

_venv-setup
source $(dirname ${BASH_SOURCE})/venv/bin/activate

if [[ ${0} == ${BASH_SOURCE} ]]; then
    # This was executed, *not* sourced. Run arguments directly.
    exec "$@"
fi
