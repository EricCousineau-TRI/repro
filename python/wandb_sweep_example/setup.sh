#!/bin/bash

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
    completion_token="2022-04-08.2"
    completion_file=${_venv_dir}/.completion-token

    if [[ -f ${completion_file} && "$(cat ${completion_file})" == "${completion_token}" ]]; then
        return 0
    fi

    set -x
    rm -rf ${_venv_dir}
    python3 -m venv ${_venv_dir}
    cd ${_venv_dir}
    ./bin/pip install -U pip wheel
    ./bin/pip install -r ${_cur_dir}/requirements.txt
    ./bin/pip freeze > ${_cur_dir}/requirements.freeze.txt

    echo "${completion_token}" > ${completion_file}
) }

_setup_venv && source ${_venv_dir}/bin/activate

export WANDB_DIR=/tmp
export PYTHONPATH=${_cur_dir}/..:${PYTHONPATH}

if [[ ${0} == ${BASH_SOURCE} ]]; then
    # This was executed, *not* sourced. Run arguments directly.
    set -eux
    env
    exec "$@"
fi
