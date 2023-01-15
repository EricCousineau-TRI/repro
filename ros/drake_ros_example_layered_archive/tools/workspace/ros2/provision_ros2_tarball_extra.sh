#!/bin/bash
source ~/.bash_aliases
set -eux

cat >&2 <<EOF
This is currently not yet reproducible. However, it serves as a simple
form of record for commands.
EOF
exit 1

script_dir=$(cd $(dirname ${BASH_SOURCE}) && pwd)

archive=ros2-humble-linux-jammy-amd64-ci.tar.bz2
cd ~/tmp/ros2_build

if [[ ! -f ${archive} ]]; then
    cd ${script_dir}
    anzu-data download --symlink --force ${archive}
    cp ${archive} ~/tmp/ros2_build
fi

if [[ ! -f ros2.repos ]]; then
    wget https://raw.githubusercontent.com/ros2/ros2/humble/ros2.repos
fi

cd ~/tmp/ros2_build
if [[ ! -d ./ros2-linux ]]; then
    tar xfj ${archive} -C .
fi

base=${PWD}/ros2-linux
set +ux
export COLCON_CURRENT_PREFIX=${base}
source ${base}/setup.sh
set -ux

# Ensure we're good.
which ros2

# Make workspace.
mkdir -p ws/src
cd ws/src
if [[ ! -d vision_opencv ]]; then
    git clone https://github.com/ros-perception/vision_opencv -b humble
fi
cd ..

rosdep update
rosdep install --from-paths src -ryi

colcon build
colcon test
colcon test-result --verbose

cd ..

cur_dir=${PWD}
# Tar up directories only.
( cd ws/install && tar cf ${cur_dir}/jammy-vision_opencv.tar */ )
