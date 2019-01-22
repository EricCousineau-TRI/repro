#!/bin/bash

_env_dir=$(cd $(dirname ${BASH_SOURCE}) && pwd)/virtualenv
_python_bin=$(which python2)
if [[ ! -f ${_env_dir}/bin/python2 ]]; then
(
    set -eux
    cd $(dirname ${_env_dir})
    ${_python_bin} -m virtualenv --python ${_python_bin} ${_env_dir}
    set +eux
    source ${_env_dir}/bin/activate
    # Install some (if not all) needed dependencies.
    pip install -r ${_env_dir}/../requirements.txt
    set -eux
)
fi

source ${_env_dir}/bin/activate
unset _env_dir _python_bin
