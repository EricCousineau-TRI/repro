#!/bin/bash

set -eux
# [[ -d build ]] && rm -rf build/

mkdir -p build && cd build
drake_install=$PWD/drake_install
mkdir -p ${drake_install}
shambhala_install=$PWD/shambhala_install
mkdir -p ${shambhala_install}

# git clone https://github.com/RobotLocomotion/drake.git
(
    cd drake
    bazel run //:install -- ${drake_install}
)

# git clone https://github.com/stonier/drake-shambhala -b stonier/link_paths
(
    cd drake-shambhala/
    mkdir -p drake_cmake_installed-build && cd drake_cmake_installed-build
    cmake ../drake_cmake_installed -Ddrake_DIR=${drake_install}/lib/cmake/drake \
        -DCMAKE_INSTALL_PREFIX=${shambhala_install}
    make -j7 install
)

ldd ${drake_install}//share/drake/examples/kuka_iiwa_arm/kuka_simulation | grep 'libdrake.so'
ldd ${shambhala_install}/lib/libparticles.so | grep 'libdrake.so'
