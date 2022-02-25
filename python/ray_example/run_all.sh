#!/bin/bash
set -eux

cd $(dirname ${BASH_SOURCE})

# ./setup.sh ./simple_job.py --local

sed \
    -E 's#\$PWD#'${PWD}'#g' \
    ./cluster.template.yaml > ./cluster.yaml

./setup.sh ray up ./cluster.yaml

./setup.sh ray exec --tmux ./cluster.yaml \
    'cd ~/ray_stuff; ./setup.sh ./simple_job.py'

cat <<EOF
To monitor:
    ./setup.sh ray monitor ./cluster.yaml

To shutdown:
    ./setup.sh ray down ./cluster.yaml
EOF
