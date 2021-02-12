#!/bin/bash

_cur_dir=$(cd $(dirname ${BASH_SOURCE}) && pwd)
_venv_dir=${_cur_dir}/venv

_setup_venv() { (
    set -eu

    completion_token="2021-02-12.0"
    completion_file=${_venv_dir}/.completion-token

    cd ${_cur_dir}

    if [[ -f ${completion_file} && $(cat ${completion_file}) == ${completion_token} ]]; then
        return 0
    fi

    set -x

    python3 -m venv ${_venv_dir}
    cd ${_venv_dir}
    ./bin/pip install -U pip wheel

    cat > ./requirements.txt <<EOF
    # For ease of use.
    numpy==1.19.5
    opencv-contrib-python==3.4.0.14
EOF

    ./bin/pip install -I -r ./requirements.txt
    ./bin/pip freeze > ${_cur_dir}/freeze.txt

    echo ${completion_token} > ${completion_file}
) }

_setup_venv && source ${_venv_dir}/bin/activate

if [[ ${0} == ${BASH_SOURCE} ]]; then
    # This was executed, *not* sourced. Run arguments directly.
    exec "$@"
fi
