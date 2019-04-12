#!/bin/bash
set -eux -o pipefail
cd $(dirname $0)

if [[ ! -d ./venv ]]; then
    python2 -m virtualenv -p python2 --system-site-packages ./venv
    ./venv/bin/pip install jupyter tqdm
fi
