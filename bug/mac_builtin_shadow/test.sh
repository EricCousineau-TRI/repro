#!/bin/bash
set -eux

# Purpose: Reproduce https://github.com/RobotLocomotion/drake/issues/8041

cd $(dirname $0)

rm -rf ./build
mkdir build
cd build

mkdir example_module
cd example_module

touch __init__.py

cat > math.py <<EOF
stuff = 1
EOF

cat > other.py <<EOF
from __future__ import absolute_import

from math import log
from .math import stuff
EOF

cat > import_test.py <<EOF
from example_module.other import log, stuff
print(log, stuff)
EOF

cd ..
export PYTHONPATH=${PWD}:${PYTHONPATH}

set +e

python example_module/import_test.py

cp example_module/import_test.py .
python import_test.py
