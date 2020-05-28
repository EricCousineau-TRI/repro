#!/bin/bash
set -eux -o pipefail

cd /drake
python3 -m virtualenv -p python3 . --system-site-packages
./bin/pip install torch==1.0.0
./bin/python - <<'EOF'
import torch
from pydrake.examples.acrobot import AcrobotPlant
acrobot = AcrobotPlant()
EOF
