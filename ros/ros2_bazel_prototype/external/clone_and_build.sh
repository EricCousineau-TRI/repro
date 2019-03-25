#!/bin/bash
set -eux -o pipefail

cd $(dirname $BASH_SOURCE)
source ./vars.sh

rmkcd() { rm -rf ${1} && mkdir -p ${1} && cd ${1}; }

# To hack with stuff.
do-overlay() { (
    rmkcd overlay_ws
    rmkcd src

    git clone https://github.com/ros2/rmw_implementation -b crystal
    (
        cd rmw_implementation && git apply < ${_cur}/patches/rmw_hack_discovery.patch
    )

    git clone https://github.com/ros2/rmw_fastrtps -b crystal

    cd ..
    # TODO: Dunno which variables matter here...
    # env PYTHONPATH=${_ros_pylib} RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    #     cmake <dir> \
    #         -DCMAKE_PREFIX_PATH=${_ros} \
    #         -DCMAKE_INSTALL_PREFIX=${_overlay}
    set +eux
    source ${_ros}/setup.bash  # :(
    set -eux

    colcon build --merge-install --symlink-install
) }

# To anchor usages.
do-examples() { (
    rmkcd examples_ws
    rmkcd src

    git clone https://github.com/ros2/examples -b 0.6.2

    rmkcd ../build
    env PYTHONPATH=${_ros_pylib} \
        cmake ../src/examples/rclcpp/minimal_publisher \
            -DCMAKE_PREFIX_PATH="${_overlay};${_ros}" \
            -DCMAKE_INSTALL_PREFIX=../install
    make install
) }

do-overlay
do-examples
