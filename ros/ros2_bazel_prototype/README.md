# `ros2_bazel_prototype`

Purpose: See what options there are to consume ROS2 C++ and Python packages
from Bazel.

## Status

**Working** - can use RPATH linked stuff (with some overlays)

## Motivation

[drake](https://drake.mit.edu) and some teams at TRI use Bazel.

For ROS1, we used `pkg-config` to get the necessary C++ build bits; for Python,
we just used the existing FHS install tree.

For ROS2, though, `pkg-config` files are no longer emitted.

## Existing

There exists [colcon-bazel](https://github.com/colcon/colcon-bazel) as a
plugin, but it seems:
* To be geared towards dispatching commands (`build`, `run`, `test`),
not for emitting artifacts for Bazel to consume other `colcon`-built
packages.
* Not to provide Bazel / Skylark rules to emit artifacts for other build
systems to use.

## This Stuff

This makes a `repository_rule` that permits extracting parameters from CMake
target (with the option for overlays).

UPDATE: `rules_foreign_cc` would be uber nice to use, but I dunno how to
consume modern CMake interface targets, or rather get the headers and libs. For
this reason, I kludged `cmake_cc.bzl`.

## Example Usage

Install ROS Crystal base stuff:
https://index.ros.org/doc/ros2/Installation/Linux-Install-Debians/

Then run:

```sh
# Build overlay necessary for RPATH stuff
./external/docker/build_overlay_archive.py
# Run a publisher
bazel run //:pub_cc
```

## Relevant Issues

*   https://github.com/ros2/rcutils/issues/143
*   https://github.com/ros2/rmw_implementation/issues/58
