#!/bin/bash

if [[ ${0} != ${BASH_SOURCE} ]]; then
    # Sourced in shell / script.
    # N.B.: Passing `-h` or `--help` in argv makes `setup.bash` choke.
    source /opt/ros/melodic/setup.bash
    unset _is_executed
else
    # Executed as binary.
    # Copied from minimal sourcing of `setup.bash`, but fixing all values,
    # and removing the reference(s) to /usr/local.
    export \
        CMAKE_PREFIX_PATH=/opt/ros/melodic \
        LD_LIBRARY_PATH=/opt/ros/melodic/lib \
        PATH=/opt/ros/melodic/bin:/usr/bin:/bin \
        PKG_CONFIG_PATH=/opt/ros/melodic/lib/pkgconfig \
        PYTHONPATH=/opt/ros/melodic/lib/python2.7/dist-packages \
        ROSLISP_PACKAGE_DIRECTORIES= \
        ROS_DISTRO=melodic \
        ROS_ETC_DIR=/opt/ros/melodic/etc/ros \
        ROS_MASTER_URI=http://localhost:11311 \
        ROS_PACKAGE_PATH=/opt/ros/melodic/share \
        ROS_PYTHON_VERSION=2 \
        ROS_ROOT=/opt/ros/melodic/share/ros \
        ROS_VERSION=1
    exec "$@"
fi
