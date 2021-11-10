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

_download_drake() { (
    # TODO(eric): Use pip once drake#16069 is resolved.

    # See: https://drake.mit.edu/from_binary.html
    # Download and echo path to stdout for capture.
    set -eux

    # v0.35.0
    base=drake-20211021-bionic.tar.gz
    dir=~/Downloads
    uri=https://drake-packages.csail.mit.edu/drake/nightly
    if [[ ! -f ${dir}/${base} ]]; then
        wget ${uri}/${base} -O ${dir}/${base}
    fi
    echo ${dir}/${base}
) }

_setup_venv() { (
    set -eu
    cd ${_cur_dir}
    completion_token="2021-11-10.1"
    completion_file=${_venv_dir}/.completion-token

    if [[ -f ${completion_file} && "$(cat ${completion_file})" == "${completion_token}" ]]; then
        return 0
    fi

    set -x
    rm -rf ${_venv_dir}

    mkdir -p ${_venv_dir}
    tar -xzf $(_download_drake) -C ${_venv_dir} --strip-components=1

    # See: https://drake.mit.edu/python_bindings.html#inside-virtualenv
    python3 -m venv ${_venv_dir} --system-site-packages
    cd ${_venv_dir}
    ./bin/pip install -U pip wheel
    ./bin/pip install --force-reinstall -r ${_cur_dir}/requirements.txt
    ./bin/pip freeze > ${_cur_dir}/requirements.freeze.txt

    echo "${completion_token}" > ${completion_file}
) }

_setup_venv \
    && source ${_venv_dir}/bin/activate \
    && export PYTHONPATH=${_cur_dir}:${PYTHONPATH}

if [[ ${0} == ${BASH_SOURCE} ]]; then
    # This was executed, *not* sourced. Run arguments directly.
    exec "$@"
fi
