#!/bin/bash
set -eux

# Installs dependencies.

cur_dir=$(cd $(dirname ${BASH_SOURCE}) && pwd)
cd ${cur_dir}

if [[ ! -d .venv-uv/ ]]; then
  ./install_uv.sh
fi
if [[ ! -d .venv/ ]]; then
  .venv-uv/bin/uv venv .venv/
fi
set +x
source ./activate.sh
set -x
.venv-uv/bin/uv pip sync ./requirements.txt
