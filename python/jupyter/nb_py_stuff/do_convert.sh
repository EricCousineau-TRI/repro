#!/bin/bash
set -eux -o pipefail

cd $(dirname $0)

./convert.py --from json --to py_v3 ./notebooks/test.{ipynb,py}
./convert.py --from py_v3 --to json ./notebooks/test.{py,roundtrip.ipynb}
./convert.py --from json --to py_v3 ./notebooks/test.roundtrip.{ipynb,py}
./convert.py --from py_v3 --to json ./notebooks/test.{roundtrip.py,final.ipynb}

# Convert both roundtrips.
git diff --no-index ./notebooks/test{,.roundtrip}.py || :
git diff --no-index ./notebooks/test{.roundtrip,.final}.ipynb || :
