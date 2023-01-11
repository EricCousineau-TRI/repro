#!/bin/bash
set -eux

set +ux
source /opt/ros/humble/setup.bash
set -ux

cd $(dirname ${BASH_SOURCE})
colcon build

if [[ -L install/my_pkg/share/my_pkg/package.xml ]]; then
    echo "Bad! We want copy of file, not symlink"
else
    echo "Good"
fi
