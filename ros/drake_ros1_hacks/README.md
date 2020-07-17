# Unofficial Hacky Example of Drake + ROS1

For official ROS1 support, see / comment on:
https://github.com/RobotLocomotion/drake/issues/9500

Only tried out on Ubuntu 18.04, ROS1 Melodic, Python 3.6. This is Python-only,
no C++.

This has no "real" build system. Only hacks to make to the minimal amount of
ROS1 Melodic's Python 2 implementation work in Python 3.6 on Ubuntu.s

## Features

* Rviz Visualization using `QueryObject`

## Setup

Make sure you have ROS1 Melodic installed. Here's a command (copy parentheses
too) for 18.04 that should make it work:

```sh
( set -eux;
    if [[ ! -f /etc/apt/sources.list.d/ros-latest.list ]]; then
      echo "Adding ROS APT repository ..."
      echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list
      sudo apt update
    fi
    sudo apt install $(
        echo ros-melodic-ros-base
        echo ros-melodic-tf2-msgs
        echo ros-melodic-tf2-py  # Requires Python 3 rebuild in `ros_py`.
        echo ros-melodic-visualization-msgs
        echo ros-melodic-rviz
    )
)
```

**TODO(eric.cousineau)**: Use Noetic for real Python 3 support.

Then source setup:

    source ./setup.sh

(If you need Drake prereqs, earlier parts of the output of `./setup.sh` will
show you how.)

This is a super hacky thing, but it's minimal, and adds stuff in `./venv`. This
procedure hacks stuff from:
https://drake.mit.edu/python_bindings.html#inside-virtualenv

It requires the sutff from `../ros1_py3_rebuild` to work (i.e. Docker).

## Rviz Visualization

Takes code from / modeled after:

* [@calderpg-tri](https://github.com/calderpg-tri)'s SceneGraph code in TRI
Anzu (private)
* [@gizatt](https://github.com/gizatt/)'s code in spartan: <br/>
<https://github.com/RobotLocomotion/spartan/blob/854b26e3a/src/catkin_projects/drake_iiwa_sim/src/ros_scene_graph_visualizer.cc>
* [@RussTedrake](https://github.com/RussTedrake/)'s `MeshcatVisualizer` in
Drake: <br/>
<https://github.com/RobotLocomotion/drake/blob/6eabb61a/bindings/pydrake/systems/meshcat_visualizer.py>

**Note**: This tries to do the "proper" approach by using `SceneGraph` /
`QueryObject`, not using `PoseBundle` nor hacks to try and use
`DrakeLcm("memq://")` to get information from LCM messages.

### Related Discussions

* [drake#9500 comment](https://github.com/RobotLocomotion/drake/issues/9500#issuecomment-620722987) - this is a new comment (ROS1 Rviz example)
in an old issue (ROS1 in Drake).
* [drake#10482 comment](https://github.com/RobotLocomotion/drake/issues/10482#issuecomment-620724731) - proper visualization workflow
* [EricCousineau-TRI/repro#1](
https://github.com/EricCousineau-TRI/repro/pull/1) - code discussion

### Known Issues

* Markers do not always show up in Rviz on the first time. I (Eric) have to
relaunch this multiple times.
* Changes in geometry are explicitly disabled here.

### Demo

Then to run, launch both Drake Visualizer and Rviz:

    # Terminal 1
    ./venv/bin/drake-visualizer

    # Terminal 2
    ./setup.sh roscore

    # Terminal 2
    ./setup.sh rviz -d ./rviz_demo.rviz

Then run the demo (pass `-h` to look at options):

    ./setup.sh python3 ./rviz_demo.py

Here's what `--single_shot` looks like in `drake-visualizer` and `rviz`:

<img src="./doc/drake_visualizer.png" height="250px"/> <img src="./doc/rviz.png" height="250px"/>
