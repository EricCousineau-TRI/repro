#!/bin/bash
set -eux -o pipefail
cd $(dirname $0)

if [[ ! -d ./venv ]]; then
    python3 -m virtualenv ./venv
    ./venv/bin/pip install jupyterlab tqdm
fi
