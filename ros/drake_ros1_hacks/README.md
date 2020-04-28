# Unofficial example of Drake + ROS1: `drake_ros1_hacks`

For official ROS1 support, see / comment on:
https://github.com/RobotLocomotion/drake/issues/9500

Only tried out on Ubuntu 18.04, ROS1 Melodic, Python 3.6.

This has no "real" build system. Only hacks to make to the minimal amount of
ROS1 Melodic's Python 2 implementation work in Python 3.6 on Ubuntu.s

## Setup

Make sure you have ROS1 Melodic installed.

Then source setup:

    source ./setup.sh

This is a super hacky thing, but it's minimal, and adds stuff in `./venv`.

## Demo

Then to run, launch both Drake Visualizer and Rviz:

    # Terminal 1
    ./venv/bin/drake-visualizer

    # Terminal 2
    ./setup.sh roscore

    # Terminal 2
    ./setup.sh rviz -d ./rviz_demo.rviz

Then run the demo:

    ./setup.sh python3 ./rviz_demo.py
