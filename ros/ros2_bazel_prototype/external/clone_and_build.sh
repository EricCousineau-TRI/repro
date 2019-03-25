#!/bin/bash
set -eux -o pipefail

cd $(dirname $BASH_SOURCE)
source ./vars.sh

# To hack with stuff.
do-overlay() { (
    mkcd -p overlay_ws/src

    (
        git clone https://github.com/ros2/rmw_implementation -b crystal && cd rmw_implementation
        git apply < ${_cur}/patches/rmw_hack_discovery.patch
    )

    mkcd -p ../build
    # TODO: Dunno which variables matter here...
    # env PYTHONPATH=${_ros_pylib} RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    #     cmake .. \
    #         -DCMAKE_PREFIX_PATH=${_ros} \
    #         -DCMAKE_INSTALL_PREFIX=${_overlay}
    set +eux
    source ${_ros}/setup.bash  # :(
    set -eux
    cmake ../src/rmw_implementation/rmw_implementation \
        -DCMAKE_PREFIX_PATH=${_ros} \
        -DCMAKE_INSTALL_PREFIX=${_overlay}
    make install
) }

# To anchor usages.
do-examples() { (
    mkcd -p examples_ws/src

    git clone https://github.com/ros2/examples -b 0.6.2

    mkcd -p ../build
    env PYTHONPATH=${_ros_pylib} \
        cmake ../src/examples/rclcpp/minimal_publisher \
            -DCMAKE_PREFIX_PATH="${_overlay};${_ros}"
    make publisher_lambda
) }

do-overlay
do-examples
