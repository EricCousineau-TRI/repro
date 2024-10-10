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
_venv_dir=${_cur_dir}/.venv

_setup_venv() { (
    set -eu

    cd ${_cur_dir}

    if [[ ! -d ${_venv_dir} ]]; then
        uv venv ${_venv_dir}
    fi

    if [[ ! -f ./requirements.txt ]]; then
        ./upgrade.sh
    fi
    uv pip sync ./requirements.txt
) }

_fg_dir=/tmp/FlameGraph

_setup_flamegraph() { (
    set -eu
    if [[ ! -d ${_fg_dir} ]]; then
        git clone https://github.com/brendangregg/FlameGraph ${_fg_dir}
    fi
    cat > ${_venv_dir}/bin/flamegraph <<'EOF'
#!/bin/bash
set -eu
/tmp/FlameGraph/flamegraph.pl "$@"
EOF
    chmod +x ${_venv_dir}/bin/flamegraph
) }

_setup_venv && source ${_venv_dir}/bin/activate
_setup_flamegraph

if [[ ${0} == ${BASH_SOURCE} ]]; then
    # This was executed, *not* sourced. Run arguments directly.
    exec "$@"
fi
