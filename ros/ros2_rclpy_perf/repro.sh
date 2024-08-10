#!/bin/bash
set -e

source /opt/ros/humble/setup.bash
set -ux

grep 'model name' /proc/cpuinfo | uniq
nproc

python3 -c 'import rclpy; print(rclpy.utilities.get_rmw_implementation_identifier())'

./pub_sub.py --count 1
./pub_sub.py --count 5
./pub_sub.py --count 10
./pub_sub.py --count 15
