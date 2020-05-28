#!/bin/bash
set -eux -o pipefail

cd $(dirname $0)
curl -o drake.tar.gz https://drake-packages.csail.mit.edu/drake/nightly/drake-20190921-bionic.tar.gz
docker build -t drake:binary .
docker run -t -v ${PWD}:/mnt --rm drake:binary bash -c /mnt/repro_container.sh
