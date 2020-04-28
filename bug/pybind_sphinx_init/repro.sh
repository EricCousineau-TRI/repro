#/bin/bash
set -eux

cd $(dirname $0)
src=${PWD}
pybind_src=${src}/../../externals/pybind11

build=${src}/build
# rm -rf ${build}
mkdir -p ${build} && cd ${build}
install=${build}/install

# Build pybind11
(
    mkdir -p pybind && cd pybind
    cmake ${pybind_src} -DCMAKE_INSTALL_PREFIX=${install} -DPYBIND11_TEST=OFF
    make install
)

# Build stuff
(
    mkdir -p pybind_sphinx_init && cd pybind_sphinx_init
    cmake ${src} -DCMAKE_PREFIX_PATH=${install} -DPYTHON_EXECUTABLE=$(which python3)
    make example  # could've added a dep, but meh
    make build_sphinx
)
