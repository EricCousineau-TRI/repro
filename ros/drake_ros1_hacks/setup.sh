#!/bin/bash

# Hacks to make things work.

_cur_dir=$(cd $(dirname ${BASH_SOURCE}) && pwd)
_venv_dir=${_cur_dir}/venv

_venv-create-if-needed()
{ (
    # Use subshell to permit errors.
    set -eu
    if [[ ! -f ${_venv_dir}/bin/activate ]]; then
        mkdir -p ${_venv_dir}
        cd ${_venv_dir}
        # file=~/Downloads/drake-20200426-bionic.tar.gz
        # test -f ${file} || wget -O ${file} https://drake-packages.csail.mit.edu/drake/nightly/$(basename ${file})
        # # https://drake.mit.edu/python_bindings.html#inside-virtualenv
        # tar xfz ${file} -C ${_venv_dir} --strip-components=1

        # Need to build using this PR:
        # https://github.com/robotlocomotion/drake/pull/13156
        cp -r /tmp/drake/* ${_venv_dir}

        python3 -m virtualenv -p python3 --system-site-packages ${_venv_dir}

        py3_pkg_dir=${_venv_dir}/lib/python3.6/site-packages
        ros_py2_pkg_dir=/usr/lib/python2.7/dist-packages

        # If this fails, we should fail fast...
        test -d ${py3_pkg_dir}

        # Hack.
        ln -s ${ros_py2_pkg_dir}/rospkg ${py3_pkg_dir}/

        py3_rebuild_dir=$(dirname ${_cur_dir})/ros1_py3_rebuild
        py3_rebuild_tar=${py3_rebuild_dir}/py3_rebuild.tar.gz
        if [[ ! -f ${py3_rebuild_tar} ]]; then
            ${py3_rebuild_dir}/do_py3_rebuild.py
        fi
        tar xfz ${py3_rebuild_tar} -C ${_venv_dir}/lib/python3.6/site-packages/
    fi
) }

_venv-source() {
    # No subshell.
    source /opt/ros/melodic/setup.bash
    # source ${_venv_dir}/bin/activate
    export PYTHONPATH=$(dirname ${_cur_dir}):${_venv_dir}/lib/python3.6/site-packages:${PYTHONPATH}
}

if [[ ${0} != ${BASH_SOURCE} ]]; then
    # Sourced in shell / script.
    _venv-create-if-needed
    _venv-source
else
    set -eu
    # Executed as binary.
    _venv-create-if-needed
    set +u
    _venv-source
    set -x
    exec "$@"
fi
