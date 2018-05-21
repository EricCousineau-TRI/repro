#!/bin/bash

cd $(dirname $0)

source /opt/ros/kinetic/setup.bash
catkin build
