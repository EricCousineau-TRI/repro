#!/bin/bash
set -eux -o pipefail

cd $(dirname $BASH_SOURCE)
source ./vars.sh

# To hack with stuff.
do-overlay() { (
    # See: https://github.com/ros2/rcpputils/issues/3
    cd overlay_ws/src
    ( set +e; cd rmw_implementation && git apply < ${_cur}/patches/rmw_hack_discovery.patch )

    cd ..
    # TODO: Dunno which variables matter here...
    # env PYTHONPATH=${_ros_pylib} RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    #     cmake <dir> \
    #         -DCMAKE_PREFIX_PATH=${_ros} \
    #         -DCMAKE_INSTALL_PREFIX=${_overlay}
    set +eux
    source ${_ros}/setup.bash  # :(
    set -eux

    colcon build --merge-install --symlink-install \
        --cmake-args -DBUILD_TESTING=OFF
) }

# To anchor usages.
do-examples() { (
    mkcd -p examples_ws/src

    git clone https://github.com/ros2/examples -b 0.6.2

    mkcd ../build
    env PYTHONPATH=${_ros_pylib} \
        cmake ../src/examples/rclcpp/minimal_publisher \
            -DCMAKE_PREFIX_PATH="${_overlay};${_ros}" \
            -DCMAKE_INSTALL_PREFIX=../install
    make install
) }

do-overlay
do-examples
