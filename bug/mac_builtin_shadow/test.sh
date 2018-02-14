#!/bin/bash
set -eux

# Purpose: Reproduce https://github.com/RobotLocomotion/drake/issues/8041

cd $(dirname $0)

rm -rf ./tmp
mkdir tmp
cd tmp

(
    mkdir test_module
    cd test_module
    touch __init__.py
    cat > math.py <<EOF
stuff = 1
EOF
    cat > other.py <<EOF
from __future__ import absolute_import

from math import log
from .math import stuff
EOF
)

cat > test_import.py <<EOF
from test_module.other import log, stuff
print(log, stuff)
EOF

python test_import.py
