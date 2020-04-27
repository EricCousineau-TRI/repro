#!/bin/bash

_venv_dir=$(cd $(dirname ${BASH_SOURCE}) && pwd)/venv
(
    set -eux
    if [[ ! -f ${_venv_dir}/bin/activate ]]; then
        mkdir -p ${_venv_dir}
        cd ${_venv_dir}
        file=~/Downloads/drake-20200426-bionic.tar.gz
        test -f ${file} || wget -O ${file} https://drake-packages.csail.mit.edu/drake/nightly/$(basename ${file})
        # https://drake.mit.edu/python_bindings.html#inside-virtualenv
        tar xfz ${file} -C ${_venv_dir} --strip-components=1
        python3 -m virtualenv -p python3 --system-site-packages ${_venv_dir}
    fi
)

source /opt/ros/melodic/setup.bash
source ${_venv_dir}/bin/activate
