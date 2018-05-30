#!/bin/bash
set -eu

cd $(dirname $0)
rm -rf build
mkdir build && cd build

drake=drake-20180529-xenial.tar.gz
drake_path=~/Downloads/${drake}
if [[ ! -f ${drake_path} ]]; then
    wget https://drake-packages.csail.mit.edu/drake/nightly/${drake} -O ${drake_path}
fi
tar xfz ${drake_path}

git clone https://github.com/RussTedrake/underactuated

mkdir pip_deps
cd pip_deps
python -m virtualenv .

set +e +u
source bin/activate
set -eu
pip install \
    matplotlib \
    meshcat \
    jupyter==1 \
    scipy
