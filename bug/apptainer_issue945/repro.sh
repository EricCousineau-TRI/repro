#!/bin/bash
set -eux

cd $(dirname ${BASH_SOURCE})
cat > ./repro.Apptainer <<'EOF'
Bootstrap: docker
From: ubuntu:20.04

%post
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get -y install mesa-utils nvidia-utils-520
EOF

apptainer build --fakeroot --sandbox ./repro.sandbox ./repro.Apptainer
apptainer exec --writable --nv ./repro.sandbox glxgears
