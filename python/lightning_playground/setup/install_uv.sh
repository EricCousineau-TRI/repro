#!/bin/bash
set -eu

# Installs uv, a faster / more comprehensive package manager than `pip`.

cur_dir=$(cd $(dirname ${BASH_SOURCE}) && pwd)
cd ${cur_dir}

python3 -m venv .venv-uv/
.venv-uv/bin/pip install uv
