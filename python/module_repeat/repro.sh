#!/bin/bash
set -eux -o pipefail

cd $(dirname $0)
# N.B. We must extend the PATH to also expose `my_submodule`.
# Normally, you should NOT do this, for the reasons shown here.
env PYTHONPATH=${PWD}/my_package python3 -m my_package.example_main

# Output
<<'EOF'
case1.__name__: my_package.my_submodule
case2.__name__: my_package.my_submodule
case3.__name__: my_submodule
case1.x: 123
case2.x: 123
case3.x: 10 (BAD)
EOF
