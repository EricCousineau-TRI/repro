#!/bin/bash

_env_dir=$(cd $(dirname ${BASH_SOURCE}) && pwd)/virtualenv
if [[ ! -f ${_env_dir}/bin/python3 ]]; then
(
    set -eux
    cd $(dirname ${_env_dir})
    python3_bin=$(which python3)
    ${python3_bin} -m virtualenv --python ${python3_bin} ${_env_dir}
    set +eux
    source ${_env_dir}/bin/activate
    # Install some (if not all) needed dependencies.
    pip install -r ${_env_dir}/../requirements.txt
    set -eux
)
fi

source ${_env_dir}/bin/activate
unset _env_dir
