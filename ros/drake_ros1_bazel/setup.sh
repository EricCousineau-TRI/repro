#!/bin/bash

# Hacks to make things work.
# Would not recommned relying on this.

_cur_dir=$(cd $(dirname ${BASH_SOURCE}) && pwd)
_venv_dir=${_cur_dir}/venv
(
    set -eu
    if [[ ! -f ${_venv_dir}/bin/activate ]]; then
        mkdir -p ${_venv_dir}
        cd ${_venv_dir}
        file=~/Downloads/drake-20200426-bionic.tar.gz
        test -f ${file} || wget -O ${file} https://drake-packages.csail.mit.edu/drake/nightly/$(basename ${file})
        # https://drake.mit.edu/python_bindings.html#inside-virtualenv
        tar xfz ${file} -C ${_venv_dir} --strip-components=1
        python3 -m virtualenv -p python3 --system-site-packages ${_venv_dir}

        # If this fails, we should fail fast...
        test -d ${_venv_dir}/lib/python3.6

        # Hack.
        ln -s /usr/lib/python2.7/dist-packages/rospkg ${_venv_dir}/lib/python3.6/site-packages

        py3_rebuild_dir=$(dirname ${_cur_dir})/ros1_py3_rebuild
        py3_rebuild_tar=${py3_rebuild_dir}/py3_rebuild.tar.gz
        if [[ ! -f ${py3_rebuild_tar} ]]; then
            ${py3_rebuild_dir}/do_py3_rebuild.py
        fi
        tar xfz ${py3_rebuild_tar} -C ${_venv_dir}/lib/python3.6/site-packages/
    fi
)

source /opt/ros/melodic/setup.bash
# source ${_venv_dir}/bin/activate
export PYTHONPATH=${_venv_dir}/lib/python3.6/site-packages::${PYTHONPATH}
