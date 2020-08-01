#!/bin/bash

_cur_dir=$(cd $(dirname ${BASH_SOURCE}) && pwd)
_venv_dir=${_cur_dir}/venv

_setup_venv() { (
    set -eu

    completion_token="2020-07-31"
    completion_file=${_venv_dir}/.completion-token

    cd ${_cur_dir}

    if [[ -f ${completion_file} && $(cat ${completion_file}) == ${completion_token} ]]; then
        return 0
    fi

    set -x

    python3 -m virtualenv -p python3 ${_venv_dir}
    cd ${_venv_dir}

    cat > ./requirements.txt <<EOF
    # For ease of use.
    numpy
    tqdm
EOF

    ./bin/pip install -r ./requirements.txt
    echo ${completion_token} > ${completion_file}
) }

_setup_venv && source ${_venv_dir}/bin/activate

if [[ ${0} == ${BASH_SOURCE} ]]; then
    # This was executed, *not* sourced. Run arguments directly.
    exec "$@"
fi
