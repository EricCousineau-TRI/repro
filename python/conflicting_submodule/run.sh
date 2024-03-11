#!/bin/bash
set -eu -o pipefail

cur_dir=$(cd $(dirname ${BASH_SOURCE}) && pwd)

export PYTHONPATH=${cur_dir}/pkg_1:${cur_dir}/pkg_2

add-top-init() {
  echo '# Empty module.' > pkg_1/top/__init__.py
  echo '# Empty module.' > pkg_2/top/__init__.py
}
rm-top-init() {
  rm -f pkg_1/top/__init__.py
  rm -f pkg_2/top/__init__.py
}

echo "[ Without 'top/__init__.py', works ]"
rm-top-init
python ./main.py

echo

echo "[ With 'top/__init__.py', fails ]"
add-top-init
python ./main.py || true  # Should fail

rm-top-init
