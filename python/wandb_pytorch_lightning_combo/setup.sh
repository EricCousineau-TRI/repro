#!/bin/bash

_cur_dir=$(cd $(dirname ${BASH_SOURCE}) && pwd)
_venv_dir=${_cur_dir}/venv

_setup_venv() { (
    set -eu

    completion_token="2021-02-18.0"
    completion_file=${_venv_dir}/.completion-token

    cd ${_cur_dir}

    if [[ -f ${completion_file} && $(cat ${completion_file}) == ${completion_token} ]]; then
        return 0
    fi

    set -x

    python3 -m venv ${_venv_dir}
    cd ${_venv_dir}
    ./bin/pip install -U pip wheel
    ./bin/pip install -I -r ${_cur_dir}/requirements.txt
    ./bin/pip freeze > ${_cur_dir}/requirements.freeze.txt

    echo ${completion_token} > ${completion_file}
) }

_setup_venv && source ${_venv_dir}/bin/activate

if [[ ${0} == ${BASH_SOURCE} ]]; then
    # This was executed, *not* sourced. Run arguments directly.
    exec "$@"
fi
