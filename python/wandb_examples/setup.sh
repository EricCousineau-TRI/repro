#!/bin/bash

_script_dir=$(cd $(dirname ${BASH_SOURCE}) && pwd)
_venv_dir=${_script_dir}/venv

_venv-setup() { (
    set +eux
    finished_token=${_venv_dir}/.finished_token
    if [[ ! -f ${finished_token} ]]; then
        python3 -m virtualenv -p python3 ${_venv_dir}
        ${_venv_dir}/bin/pip install -r ${_script_dir}/requirements.txt
        touch ${finished_token}
    fi
) }

_venv-setup
source $(dirname ${BASH_SOURCE})/venv/bin/activate

if [[ ${0} == ${BASH_SOURCE} ]]; then
    # This was executed, *not* sourced. Run arguments directly.
    exec "$@"
fi
