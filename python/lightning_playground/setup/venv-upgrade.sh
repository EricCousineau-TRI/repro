#!/bin/bash
set -eux

cur_dir=$(cd $(dirname ${BASH_SOURCE}) && pwd)
cd ${cur_dir}

if [[ ! -d .venv-uv/ ]]; then
  ./install_uv.sh
fi
.venv-uv/bin/uv pip compile ./requirements.in --output-file /tmp/requirements.txt --generate-hashes
grep '\--find-links' ./requirements.in > /tmp/find-links.txt || true
cat /tmp/find-links.txt /tmp/requirements.txt > ./requirements.txt
