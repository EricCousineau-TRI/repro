# Joint Limit Things

## Prereqs

Tested on Ubuntu 18.04 (Bionic). Needs ROS1 Melodic, Drake prereqs, and
pyassimp.

## Setup

You can just run this:

```sh
./setup.sh
```

This will:

* Set up a small `virtualenv` with Drake and JupyterLab
* Clone `ur_description` and, uh, convert it to format that Drake can use :(

## Running

```sh
./setup.sh jupyter lab ./joint_limits.ipynb
```
