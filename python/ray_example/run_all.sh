#!/bin/bash
set -eux

cd $(dirname ${BASH_SOURCE})

set +ux
source ./setup.sh
set -ux

# # Test job script locally.
#./simple_job.py --local

sed \
    -E 's#\$PWD#'${PWD}'#g' \
    ./cluster.template.yaml > ./cluster.yaml

maybe-workaround() {
    true  #  add single command to prevent syntax error

    # Uncomment to use workaround
    ./ray_exec_all.py 'ray stop || true'
}

maybe-workaround

ray up -y ./cluster.yaml

ray exec ./cluster.yaml '~/ray_stuff/simple_job.py'

cat <<'EOF'
Get more info:
    ray exec ./cluster.yaml 'tail -n 100 -f /tmp/ray/session_latest/logs/monitor*'
EOF

# Try to show restart behavior.
# https://github.com/ray-project/ray/issues/19834#issuecomment-1054897153

maybe-workaround

ray up -y ./cluster.yaml

# This should
ray exec ./cluster.yaml '~/ray_stuff/simple_job.py'
