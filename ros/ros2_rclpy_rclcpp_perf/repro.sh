#!/bin/bash
set -e

source /opt/ros/humble/setup.bash
set -ux

grep 'model name' /proc/cpuinfo | uniq
nproc

./pub_sub.py --count 1
./pub_sub.py --count 5
./pub_sub.py --count 10
