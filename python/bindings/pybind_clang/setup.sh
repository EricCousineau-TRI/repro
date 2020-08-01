#!/bin/bash

_cur_dir=$(cd $(dirname ${BASH_SOURCE}) && pwd)
_venv_dir=${_cur_dir}/venv

_setup_venv() { (
    set -eu

    completion_token="2020-07-31.5"
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

    # Install later version of castxml.
    # See README here: https://github.com/CastXML/CastXMLSuperbuild/tree/75ec9ef4ad48ddab605627d783bfdee57fd7bcbf
    # This is v0.3.4 for Linux: https://data.kitware.com/#item/5ee7eb659014a6d84ec1f25c
    if [[ ! -f ./castxml.tar.gz ]]; then
        wget https://data.kitware.com/api/v1/file/5ee7eb659014a6d84ec1f25e/download -O ./castxml.tar.gz
    fi
    tar xfz ./castxml.tar.gz -C . --strip-components 1

    ./bin/pip install -I -r ./requirements.txt
    # https://gitlab.kitware.com/autopybind11/autopybind11/-/merge_requests/30
    # git+https://gitlab.kitware.com/EricCousineau-TRI/autopybind11.git@ce11831
    ./bin/pip install -I -e ~/proj/tri/repo/externals/autopybind11

    ./bin/pip freeze > ${_cur_dir}/freeze.txt

    echo ${completion_token} > ${completion_file}
) }

_setup_venv && source ${_venv_dir}/bin/activate

if [[ ${0} == ${BASH_SOURCE} ]]; then
    # This was executed, *not* sourced. Run arguments directly.
    exec "$@"
fi
