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

_download_drake() { (
    # See: https://drake.mit.edu/from_binary.html
    # Download and echo path to stdout for capture.
    set -eux

    base=drake-latest-focal.tar.gz
    dir=~/Downloads
    uri=https://drake-packages.csail.mit.edu/drake/nightly
    if [[ ! -f ${dir}/${base} ]]; then
        wget ${uri}/${base} -O ${dir}/${base}
    fi
    echo ${dir}/${base}
) }

_preprocess_sdf() { (
    #convert .stl and .dae entries to .obj
    sed -i 's/.stl/.obj/g' "$1$2"
    # Some sdfs have a comment before the xml tag
    # this makes the parser fail, since the tag is optional
    # we'll remove it as safety workaround
    sed -i '/<?xml*/d' "$1$2"

) }

_provision_repos() { (
    set -eu
    cd ${_cur_dir}
    repo_dir=${PWD}/repos
    completion_token=2021-03-12.1
    completion_file=$1/.completion-token

    if [[ "$2" == *\.sdf ]]
    then
        _preprocess_sdf "$1" "$2"
        ./render_ur_urdfs.py "$1" "$2"
    else
        if [[ -f ${completion_file} && "$(cat ${completion_file})" == "${completion_token}" ]]; then
        return 0
        fi
        set -x
        rm -rf ${repo_dir}

        mkdir ${repo_dir} && cd ${repo_dir}

        git clone https://github.com/ros-industrial/universal_robot
        cd universal_robot/
        git checkout e8234318cc94  # From melodic-devel-staging
        # Er... dunno what to do about this, so hackzzz
        cd ${_cur_dir}
        ./ros_setup.bash ./render_ur_urdfs.py "$1" "$2"
    fi

    echo "${completion_token}" > ${completion_file}
) }

_setup_venv() { (
    set -eu
    cd ${_cur_dir}
    completion_token="$(cat ./requirements.txt)"
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
    ./bin/pip install -I pip wheel
    ./bin/pip install -I -r ${_cur_dir}/requirements.txt
    ./bin/pip freeze > ${_cur_dir}/requirements.freeze.txt

    echo "${completion_token}" > ${completion_file}
) }

if [  $# -lt "2" ]
    then
    echo "Please provide path to model directory and model file name."
    echo "      Usage:"
    echo "                  $./setup.sh <model_directory_path> <model_file_name> ./[executable]"
    echo "      or"
    echo "                  $source ./setup.sh <model_directory_path> <model_file_name>"
    echo "                  $./[executable]"

    exit 1
fi

_setup_venv && source ${_venv_dir}/bin/activate

_provision_repos "$1" "$2"

if [[ ${0} == ${BASH_SOURCE} ]]; then
    # This was executed, *not* sourced. Run arguments directly.
    set -eux
    env
    exec "${@:3}"
fi
