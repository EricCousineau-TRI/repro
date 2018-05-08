#!/bin/bash
set -eux -o pipefail

cd $(dirname $0)

./convert.py --from json --to py_v3 ./notebooks/test{.ipynb,.initial.py}
./convert.py --from py_v3 --to json ./notebooks/test.initial{.py,.ipynb}
./convert.py --from json --to py_v3 ./notebooks/test{.initial.ipynb,.final.py}
./convert.py --from py_v3 --to json ./notebooks/test.final{.py,.ipynb}

# Compare both roundtrips.
(
    git diff --no-index ./notebooks/test{.initial,.final}.py
    git diff --no-index ./notebooks/test{.initial,.final}.ipynb
) || {
    echo "FAILURE" >&2
    exit 1
}
