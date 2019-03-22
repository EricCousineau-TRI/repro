#!/bin/bash
set -eux -o pipefail

# More local repro of https://bugs.python.org/issue6386

py=${1}

cd $(dirname $0)
rm -rf symlink_junk_gen && mkdir symlink_junk_gen && cd symlink_junk_gen

mkdir -p sub symlink

cat > root.py <<EOF
import sub
print(sub.func())
EOF

cat > sub.py <<EOF
def func():
    print("Hello")
EOF

ln -sr root.py ./symlink

${py} ./symlink/root.py
find . -name '*.pyc' -o -name '__pycache__'
