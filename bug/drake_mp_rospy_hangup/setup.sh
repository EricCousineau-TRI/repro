#!/bin/bash

# Set up things for prefix
fhs-extend()
{
    local python_version=2.7
    local prefix=${1%/}
    export-prepend PYTHONPATH $prefix/lib:$prefix/lib/python${python_version}/dist-packages:$prefix/lib/python${python_version}/site-packages
    export-prepend PATH $prefix/bin
    export-prepend LD_LIBRARY_PATH $prefix/lib
    export-prepend PKG_CONFIG_PATH $prefix/lib/pkgconfig:$prefix/share/pkgconfig
    echo "[ FHS Environment extended: ${prefix} ]"
}
export-prepend () 
{ 
    eval "export $1=\"$2:\$$1\""
}

_cur=$(cd $(dirname ${BASH_SOURCE}) && pwd)
(
    set -eux
    build=${_cur}/venv
    mkdir -p ${build}
    cd ${build}
    file=drake-20190122-bionic.tar.gz
    test -f ${file} || wget https://drake-packages.csail.mit.edu/drake/nightly/${file}
    test -d drake || tar xfz ${file}
)

source /opt/ros/melodic/setup.bash
fhs-extend ${_cur}/venv/drake
