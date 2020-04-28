# Unofficial example of Drake + ROS1 in Bazel

For official support, see / comment on:
https://github.com/RobotLocomotion/drake/issues/9500

Only tried out on Ubuntu 18.04, ROS1 Melodic.

## Setup

Make sure you have ROS1 Melodic installed.

Then source setup:

    source ./setup.sh

This is a super hacky thing, but it's minimal, and adds stuff in `./venv`.

Then to run, launch both Drake Visualizer and RViz:

    # Terminal 1
    ./venv/bin/drake-visualizer

    # Terminal 2
    source ./setup.sh
    roscore

    # Terminal 2
    source ./setup.sh
    rviz -d ./demo.rviz

Then run the demo:

    source ./setup.sh
    python3 ./demo.py
