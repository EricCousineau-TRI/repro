#!/bin/bash
set -eux

set +ux
source /opt/ros/humble/setup.bash
set -ux

cd $(dirname ${BASH_SOURCE})
colcon build --symlink-install

if [[ -f $(readlink install/my_pkg/share/my_pkg/package.xml) ]]; then
    echo "Good"
else
    echo "Bad"
fi
